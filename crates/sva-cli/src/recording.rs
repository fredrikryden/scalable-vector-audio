// Concern: reads a recording file to planes at its own rate: wav exactly, mp3, aac in mp4, flac, ogg decoded | Non-concern: writing a WAV (wav.rs), resampling | IO: (path) -> planes, a rate or CliError

use std::path::Path;

use sva_core::CliError;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// One plane per channel, and the rate they were recorded at.
type Planes = (Vec<Vec<f32>>, u32);

/// Every extension a recording is read from.
pub const READS: [&str; 8] = ["wav", "mp3", "m4a", "mp4", "aac", "flac", "ogg", "oga"];

/// A recording in no format this reads, or one that does not decode.
pub const UNREADABLE: &str = sva_core::RECORDING_UNREADABLE;

/// Every channel in its own plane at the file's own rate, never resampled, so a reference
/// recording's own clock is what every reading is against.
pub fn read_recording(path: &Path) -> Result<Planes, CliError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match extension.as_str() {
        "wav" => wav(path),
        known if READS.contains(&known) => decoded(path, known),
        _ => Err(CliError::usage(
            UNREADABLE,
            format!(
                "`{}` is in no format this reads; it reads `.{}`",
                path.display(),
                READS.join("`, `.")
            ),
        )),
    }
}

/// A file that is not there is `not_found`, never the `internal_error` of a tool that broke.
fn missing(path: &Path, e: &std::io::Error) -> CliError {
    match e.kind() {
        std::io::ErrorKind::NotFound => {
            CliError::NotFound(format!("no such file: {}", path.display()))
        }
        _ => CliError::Io(format!("could not open {}: {e}", path.display())),
    }
}

fn wav(path: &Path) -> Result<Planes, CliError> {
    let mut reader = hound::WavReader::open(path).map_err(|e| match e {
        hound::Error::IoError(io) => missing(path, &io),
        e => CliError::usage(
            UNREADABLE,
            format!("could not read {}: {e}", path.display()),
        ),
    })?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(CliError::usage(
            UNREADABLE,
            format!("{} declares 0 channels", path.display()),
        ));
    }
    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let bail = |e: hound::Error| CliError::Io(format!("could not read {}: {e}", path.display()));
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, sample) in reader.samples::<f32>().enumerate() {
                planes[i % channels].push(sample.map_err(bail)?);
            }
        }
        hound::SampleFormat::Int => {
            // The full scale `wav.rs` quantizes against, `2^(b-1) - 1`.
            let full_scale = ((1i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            for (i, sample) in reader.samples::<i32>().enumerate() {
                planes[i % channels].push(sample.map_err(bail)? as f32 / full_scale);
            }
        }
    }
    Ok((planes, spec.sample_rate))
}

fn decoded(path: &Path, extension: &str) -> Result<Planes, CliError> {
    let file = std::fs::File::open(path).map_err(|e| missing(path, &e))?;
    let refused = |why: String| CliError::usage(UNREADABLE, why);
    let unread = |e: Error| refused(format!("could not decode `{}`: {e}", path.display()));
    let mut hint = Hint::new();
    hint.with_extension(extension);
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(unread)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| refused(format!("`{}` holds no audio track", path.display())))?;
    let id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| refused(format!("`{}` names no audio codec", path.display())))?;
    // An integer source reads at the wav full scale, `2^(b-1) - 1`, where the decoder's is `2^(b-1)`.
    let scale = match (extension, params.bits_per_sample) {
        ("flac", Some(bits)) if (2..=32).contains(&bits) => {
            let full = 2f64.powi(bits as i32 - 1);
            (full / (full - 1.0)) as f32
        }
        _ => 1.0,
    };
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(unread)?;
    let (mut planes, mut rate): (Vec<Vec<f32>>, Option<u32>) = (Vec::new(), None);
    let mut block: Vec<Vec<f32>> = Vec::new();
    while let Some(packet) = format.next_packet().map_err(unread)? {
        if packet.track_id != id {
            continue;
        }
        let audio = decoder.decode(&packet).map_err(unread)?;
        rate.get_or_insert(audio.spec().rate());
        audio.copy_to_vecs_planar(&mut block);
        planes.resize_with(block.len(), Vec::new);
        for (plane, part) in planes.iter_mut().zip(&block) {
            plane.extend(part.iter().map(|v| v * scale));
        }
    }
    let rate = rate.ok_or_else(|| refused(format!("`{}` holds no audio", path.display())))?;
    if matches!(extension, "m4a" | "mp4") {
        let bytes = std::fs::read(path).map_err(|e| missing(path, &e))?;
        if let Some((skip, keep)) = edit(&bytes, rate) {
            for plane in &mut planes {
                plane.drain(..skip.min(plane.len()));
                if let Some(keep) = keep {
                    plane.truncate(keep);
                }
            }
        }
    }
    Ok((planes, rate))
}

/// Each box in `data[from..to]`: its type and its body's bounds.
fn boxes(data: &[u8], from: usize, to: usize) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let mut at = from;
    while at + 8 <= to {
        let size = u32::from_be_bytes(data[at..at + 4].try_into().expect("4 bytes")) as usize;
        let kind: [u8; 4] = data[at + 4..at + 8].try_into().expect("4 bytes");
        let (body, end) = match size {
            0 => (at + 8, to),
            1 if at + 16 <= to => {
                let wide = u64::from_be_bytes(data[at + 8..at + 16].try_into().expect("8 bytes"));
                (at + 16, at.saturating_add(wide as usize))
            }
            n => (at + 8, at.saturating_add(n)),
        };
        if end > to || end < body {
            break;
        }
        out.push((kind, body, end));
        at = end;
    }
    out
}

fn word(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from(u32::from_be_bytes(
        data.get(at..at + 4)?.try_into().ok()?,
    )))
}

fn wide(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(data.get(at..at + 8)?.try_into().ok()?))
}

/// A box's timescale, after its version byte: at 12 in version 0, 20 in version 1.
fn timescale(data: &[u8], body: usize) -> Option<u64> {
    word(data, body + if data.get(body)? == &1 { 20 } else { 12 })
}

/// The frames the sound track's first edit of media skips, and keeps where it states a
/// duration: an MP4 states its encoder's priming and padding in its edit list, which the
/// decoder does not apply. An empty edit, media time -1, places no media and is passed over.
fn edit(data: &[u8], rate: u32) -> Option<(usize, Option<usize>)> {
    let find = |within: (usize, usize), kind: &[u8; 4]| {
        boxes(data, within.0, within.1)
            .into_iter()
            .find(|(k, ..)| k == kind)
            .map(|(_, b, e)| (b, e))
    };
    let moov = find((0, data.len()), b"moov")?;
    let movie = timescale(data, find(moov, b"mvhd")?.0)?;
    for (kind, body, end) in boxes(data, moov.0, moov.1) {
        if &kind != b"trak" {
            continue;
        }
        let mdia = find((body, end), b"mdia")?;
        if data.get(find(mdia, b"hdlr")?.0 + 8..)?.get(..4)? != b"soun" {
            continue;
        }
        let media = timescale(data, find(mdia, b"mdhd")?.0)?;
        let elst = find(find((body, end), b"edts")?, b"elst")?.0;
        let v1 = data.get(elst)? == &1;
        let frames = |count: u64, scale: u64| {
            (count as f64 * f64::from(rate) / scale as f64).round() as usize
        };
        for entry in 0..word(data, elst + 4)? as usize {
            let at = elst + 8 + entry * if v1 { 20 } else { 12 };
            let (duration, start) = match v1 {
                true => (wide(data, at)?, wide(data, at + 8)? as i64),
                false => (
                    word(data, at)?,
                    i64::from(word(data, at + 4)? as u32 as i32),
                ),
            };
            if let Ok(start) = u64::try_from(start) {
                let keep = (duration > 0).then(|| frames(duration, movie));
                return Some((frames(start, media), keep));
            }
        }
        return None;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::{SampleEncoding, write_channels};

    fn tmp(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sva-cli-recording-{name}-{:x}.wav",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// Inverses at 32-bit float: the round trip is exact, at the file's own rate.
    #[test]
    fn reads_round_trips_a_float_wav_exactly_at_its_own_rate() {
        let path = tmp("read-float-roundtrip");
        let left = [0.5f32, -0.25, 0.75, -1.0];
        let right = [-0.5f32, 0.25, -0.75, 1.0];
        write_channels(&[&left, &right], 48_000, &path, SampleEncoding::Float).unwrap();

        let (planes, sample_rate) = read_recording(&path).unwrap();
        assert_eq!(sample_rate, 48_000, "no resampling, ever");
        assert_eq!(planes, vec![left.to_vec(), right.to_vec()]);

        let _ = std::fs::remove_file(&path);
    }

    /// 16-bit PCM quantizes on the way in, so the round trip is within one step.
    #[test]
    fn reads_unquantizes_pcm16_within_one_step() {
        let path = tmp("read-pcm16-roundtrip");
        let pcm = [0.5f32, -0.5, 0.1, -0.9];
        write_channels(&[&pcm], 44_100, &path, SampleEncoding::Pcm(16)).unwrap();

        let (planes, sample_rate) = read_recording(&path).unwrap();
        let mono = &planes[0];
        assert_eq!(sample_rate, 44_100);
        assert_eq!(mono.len(), pcm.len());
        for (got, want) in mono.iter().zip(&pcm) {
            assert!(
                (got - want).abs() < 1.0 / f32::from(i16::MAX),
                "{got} vs {want}"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    fn write_int(path: &std::path::Path, bits_per_sample: u16, values: &[i32], sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &v in values {
            writer.write_sample(v).unwrap();
        }
        writer.finalize().unwrap();
    }

    /// hound's own `Sample for i32` reader already covers 8/16/24/32-bit int — this is the
    /// unquantize half, checked at the two depths `write_channels` never itself produces.
    #[test]
    fn reads_unquantizes_every_int_bit_depth_hound_can_decode() {
        let path = tmp("read-8bit");
        write_int(&path, 8, &[127, -128, 0], 22_050);
        let (planes, sample_rate) = read_recording(&path).unwrap();
        assert_eq!(sample_rate, 22_050);
        assert_eq!(planes.len(), 1);
        assert!((planes[0][0] - 1.0).abs() < 0.01, "{:?}", planes[0]);
        assert!((planes[0][1] + 1.0).abs() < 0.01, "{:?}", planes[0]);
        assert_eq!(planes[0][2], 0.0);
        let _ = std::fs::remove_file(&path);

        let path = tmp("read-24bit");
        let full_scale = (1i64 << 23) - 1;
        write_int(
            &path,
            24,
            &[full_scale as i32, -(full_scale as i32), 0],
            96_000,
        );
        let (planes, sample_rate) = read_recording(&path).unwrap();
        assert_eq!(sample_rate, 96_000);
        assert_eq!(planes[0], vec![1.0, -1.0, 0.0]);
        let _ = std::fs::remove_file(&path);
    }

    /// A round-robin de-interleave, checked against a file `write_channels` never produces
    /// itself: an odd number of channels with a visibly different signal on each.
    #[test]
    fn reads_de_interleaves_every_component_into_its_own_plane() {
        let path = tmp("read-three-channel");
        let a = [0.1f32, 0.2];
        let b = [0.3f32, 0.4];
        let c = [0.5f32, 0.6];
        write_channels(&[&a, &b, &c], 44_100, &path, SampleEncoding::Float).unwrap();

        let (planes, _) = read_recording(&path).unwrap();
        assert_eq!(planes, vec![a.to_vec(), b.to_vec(), c.to_vec()]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reads_reports_a_missing_file_as_not_found_not_a_panic() {
        let path = tmp("read-missing");
        assert!(matches!(read_recording(&path), Err(CliError::NotFound(_))));
    }

    /// `kind`'s box around `body`.
    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend(kind);
        out.extend(body);
        out
    }

    /// A version 0 movie at timescale 1000 holding one sound track at 48000 with `edits`,
    /// each `(duration in the movie's scale, media time in the track's)`.
    fn movie(edits: &[(u32, i32)]) -> Vec<u8> {
        let scaled = |scale: u32| [&[0u8; 12][..], &scale.to_be_bytes()].concat();
        let mut elst = vec![0, 0, 0, 0];
        elst.extend((edits.len() as u32).to_be_bytes());
        for (duration, start) in edits {
            elst.extend(duration.to_be_bytes());
            elst.extend(start.to_be_bytes());
            elst.extend([0, 1, 0, 0]);
        }
        let hdlr = [&[0u8; 8][..], b"soun"].concat();
        let mdia = [boxed(b"mdhd", &scaled(48_000)), boxed(b"hdlr", &hdlr)].concat();
        let trak = [
            boxed(b"edts", &boxed(b"elst", &elst)),
            boxed(b"mdia", &mdia),
        ]
        .concat();
        boxed(
            b"moov",
            &[boxed(b"mvhd", &scaled(1000)), boxed(b"trak", &trak)].concat(),
        )
    }

    #[test]
    fn an_edit_list_skips_the_priming_and_keeps_the_duration_it_states() {
        assert_eq!(
            edit(&movie(&[(5000, 1024)]), 48_000),
            Some((1024, Some(240_000)))
        );
        assert_eq!(
            edit(&movie(&[(5000, 1024)]), 44_100),
            Some((941, Some(220_500)))
        );
    }

    #[test]
    fn an_empty_edit_is_passed_over_and_an_unstated_duration_keeps_every_frame() {
        assert_eq!(
            edit(&movie(&[(20, -1), (5000, 2112)]), 48_000),
            Some((2112, Some(240_000)))
        );
        assert_eq!(edit(&movie(&[(0, 1024)]), 48_000), Some((1024, None)));
        assert_eq!(edit(&movie(&[(20, -1)]), 48_000), None);
    }

    #[test]
    fn a_format_it_does_not_read_refuses_naming_every_one_it_does() {
        let refused = read_recording(Path::new("take.opus")).unwrap_err();
        assert!(
            matches!(
                refused,
                CliError::Usage {
                    code: UNREADABLE,
                    ..
                }
            ),
            "{refused:?}"
        );
        for extension in READS {
            assert!(
                refused.message().contains(&format!("`.{extension}`")),
                "{refused:?}"
            );
        }
    }
}
