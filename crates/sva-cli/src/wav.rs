// Concern: writes components as one interleaved WAV, any bit depth or float | Non-concern: choosing the destination, reading one (recording.rs) | IO: (planes, sample_rate, path) -> () or CliError

use std::path::Path;

use sva_core::CliError;

/// Integer PCM at up to 16 bits is the format tag Python's stdlib `wave` module can open;
/// hound marks any `bits_per_sample > 16` format `WAVE_FORMAT_EXTENSIBLE`, which it refuses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleEncoding {
    Float,
    /// This many bits, 16 at most, left-justified in an 8- or 16-bit word.
    Pcm(u8),
}

impl SampleEncoding {
    pub fn of(bits: i32) -> SampleEncoding {
        match u8::try_from(bits) {
            Ok(bits) if bits <= 16 => SampleEncoding::Pcm(bits),
            _ => SampleEncoding::Float,
        }
    }
}

pub fn write_channels(
    planes: &[&[f32]],
    sample_rate: u32,
    path: &Path,
    encoding: SampleEncoding,
) -> Result<(), CliError> {
    match encoding {
        SampleEncoding::Float => write_as::<f32>(planes, sample_rate, path, 32, |s| s),
        SampleEncoding::Pcm(bits) if bits <= 8 => {
            write_as::<i8>(planes, sample_rate, path, 8, |s| quantize(s, bits, 8) as i8)
        }
        SampleEncoding::Pcm(bits) => write_as::<i16>(planes, sample_rate, path, 16, |s| {
            quantize(s, bits, 16) as i16
        }),
    }
}

pub fn write_wav(
    samples: &[f32],
    sample_rate: u32,
    path: &Path,
    encoding: SampleEncoding,
) -> Result<(), CliError> {
    write_channels(&[samples], sample_rate, path, encoding)
}

/// `bits` of full scale, shifted up to fill a `word`-bit sample.
pub(crate) fn quantize(sample: f32, bits: u8, word: u8) -> i32 {
    let full = ((1i32 << (bits - 1)) - 1) as f32;
    ((sample.clamp(-1.0, 1.0) * full).round() as i32) << (word - bits)
}

fn write_as<S: hound::Sample>(
    planes: &[&[f32]],
    sample_rate: u32,
    path: &Path,
    bits_per_sample: u16,
    convert: impl Fn(f32) -> S,
) -> Result<(), CliError> {
    let sample_format = if bits_per_sample == 32 {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    };
    let spec = hound::WavSpec {
        channels: planes.len().max(1) as u16,
        sample_rate,
        bits_per_sample,
        sample_format,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| CliError::Io(format!("could not create {}: {e}", path.display())))?;
    for i in 0..planes.first().map_or(0, |p| p.len()) {
        for plane in planes {
            writer.write_sample(convert(plane[i])).map_err(|e| {
                CliError::Io(format!("could not write sample to {}: {e}", path.display()))
            })?;
        }
    }
    writer
        .finalize()
        .map_err(|e| CliError::Io(format!("could not finalize {}: {e}", path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("sva-cli-wav-{name}-{:x}.wav", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn pcm16_writes_a_plain_format_tag_a_bare_riff_parser_can_read() {
        let path = tmp("pcm16-tag");
        write_wav(
            &[0.5, -0.5, 1.5, -1.5],
            44100,
            &path,
            SampleEncoding::Pcm(16),
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        let format_tag = u16::from_le_bytes([bytes[20], bytes[21]]);
        assert_ne!(
            format_tag, 0xFFFE,
            "WAVE_FORMAT_EXTENSIBLE, which Python's `wave` refuses"
        );
        assert_eq!(format_tag, 1, "WAVE_FORMAT_PCM");

        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Int);
        let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(samples, vec![16384, -16384, i16::MAX, -i16::MAX]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn float_still_carries_every_bit_and_still_reads_back_as_extensible() {
        let path = tmp("float-tag");
        let pcm = [0.125f32, -0.25, 0.75];
        write_wav(&pcm, 8000, &path, SampleEncoding::Float).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let format_tag = u16::from_le_bytes([bytes[20], bytes[21]]);
        assert_eq!(
            format_tag, 0xFFFE,
            "float always needs WAVEFORMATEXTENSIBLE in hound"
        );

        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().bits_per_sample, 32);
        let samples: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
        assert_eq!(samples, pcm);

        let _ = std::fs::remove_file(&path);
    }
}
