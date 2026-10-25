// Concern: states what the hearing curves owe their published points, their rows and between rows | Non-concern: a reading through them (src/measure/internal/) | IO: (Hz, dB) -> asserted Hz, dB, p

use sva_samples::measure::hearing::{
    C3, D_PRIME_TWO_DOWN_ONE_UP, P_FLOOR, cam, cam_hz, ear_gain_db, erb_hz, free_field_db,
    frequency_jnd_hz, level_jnd_db, middle_ear_db, p51, roex, roex_p, threshold_excitation_db,
};

fn close(got: f64, want: f64, within: f64) {
    assert!((got - want).abs() <= within, "{got} against {want}");
}

/// erf by its Maclaurin series, summed until a term no longer moves the sum.
fn erf(x: f64) -> f64 {
    let mut sum = 0.0;
    let mut power = x;
    let mut n = 0.0;
    loop {
        let term = power / (2.0 * n + 1.0);
        if sum + term == sum {
            return 2.0 / std::f64::consts::PI.sqrt() * sum;
        }
        sum += term;
        n += 1.0;
        power *= -x * x / n;
    }
}

#[test]
fn an_erb_at_one_kilohertz_is_132_4_hz_and_the_erb_number_is_its_integral() {
    close(erb_hz(1000.0), 132.4, 0.05);
    let h = 1e-3;
    let slope = (cam(1000.0 + h) - cam(1000.0 - h)) / (2.0 * h);
    close(slope * erb_hz(1000.0), 1.0, 1e-5);
    close(C3, 21.366, 5e-4);
    for z in [1.8, 15.0, 38.9] {
        close(cam(cam_hz(z)), z, 1e-12);
    }
}

#[test]
fn the_two_down_one_up_d_prime_puts_two_alternatives_at_seventy_point_seven_percent() {
    let correct = 0.5 * (1.0 + erf(D_PRIME_TWO_DOWN_ONE_UP / 2.0));
    close(correct, std::f64::consts::FRAC_1_SQRT_2, 4.0 * f64::EPSILON);
    close(D_PRIME_TWO_DOWN_ONE_UP, 0.7707, 5e-5);
}

#[test]
fn jesteadts_level_step_at_forty_db_sl_is_its_fit_over_its_criterion() {
    let ratio = 0.463 * libm::pow(10.0, -0.072 * 4.0);
    close(ratio, 0.2386, 5e-5);
    let published_db = 10.0 * libm::log10(1.0 + ratio);
    close(
        level_jnd_db(40.0) * D_PRIME_TWO_DOWN_ONE_UP,
        published_db,
        1e-12,
    );
    close(level_jnd_db(40.0), 1.206, 5e-4);
    assert!(
        level_jnd_db(80.0) < level_jnd_db(5.0),
        "the near miss to Weber"
    );
}

#[test]
fn nelsons_curve_at_one_kilohertz_forty_db_sl_is_4_17_hz_and_2_83_hz_at_unit_d_prime() {
    let wier = frequency_jnd_hz(1000.0, 40.0) * D_PRIME_TWO_DOWN_ONE_UP;
    close(wier * 1.49 / 0.78, 4.17, 5e-3);
    close(wier, 2.18, 5e-3);
    close(frequency_jnd_hz(1000.0, 40.0), 2.83, 5e-3);
}

#[test]
fn the_ear_tables_read_their_own_rows() {
    close(free_field_db(2000.0), 12.0, 0.0);
    close(free_field_db(0.0), 0.0, 0.0);
    close(free_field_db(11200.0), 1.9, 0.0);
    close(middle_ear_db(1000.0), 2.6, 0.0);
    close(middle_ear_db(0.0), 50.0, 0.0);
    close(middle_ear_db(18000.0), 46.5, 0.0);
    close(middle_ear_db(31.5), 25.85, 0.0);
    close(ear_gain_db(2000.0), 12.0 - 8.5, 1e-15);
    close(threshold_excitation_db(50.0), 28.2 - 0.58, 1e-15);
    close(threshold_excitation_db(100.0), 15.68 - 0.58, 1e-15);
    for hz in [500.0, 1000.0, 8000.0, 17000.0, 20000.0] {
        close(threshold_excitation_db(hz), 3.15, 1e-15);
    }
}

#[test]
fn between_rows_a_table_is_linear_in_hz_and_beyond_its_ends_holds_its_end_row() {
    close(free_field_db(1800.0), (6.6 + 12.0) / 2.0, 1e-15);
    close(middle_ear_db(10.0), (50.0 + 39.6) / 2.0, 1e-15);
    close(
        threshold_excitation_db(25.0),
        (32.0 + 28.2) / 2.0 - 0.58,
        1e-15,
    );
    close(free_field_db(30000.0), 2.5, 0.0);
    close(middle_ear_db(25000.0), 50.0, 0.0);
    close(threshold_excitation_db(-1.0), 32.0 - 0.58, 0.0);
}

#[test]
fn every_table_is_continuous_across_its_rows() {
    let delta = 1e-6;
    for step in 0..=50_000 {
        let hz = f64::from(step) * 0.5;
        for read in [free_field_db, middle_ear_db, threshold_excitation_db] {
            let at = read(hz);
            close(read(hz - delta), at, 1e-5);
            close(read(hz + delta), at, 1e-5);
        }
    }
}

#[test]
fn the_roex_upper_slope_ignores_level_and_the_lower_widens_with_it_down_to_its_floor() {
    let p = p51(1000.0);
    close(p, 4000.0 / erb_hz(1000.0), 1e-12);
    close(roex_p(1000.0, 1200.0, 100.0), p, 0.0);
    close(roex_p(1000.0, 1000.0, 100.0), p, 0.0);
    close(roex_p(1000.0, 800.0, 51.0), p, 0.0);
    close(roex_p(1000.0, 800.0, 61.0), p - 3.5, 1e-12);
    close(
        roex_p(500.0, 400.0, 61.0),
        p51(500.0) * (1.0 - 3.5 / p),
        1e-12,
    );
    close(roex_p(1000.0, 800.0, 1000.0), P_FLOOR, 0.0);
    close(roex(p, 1000.0, 1000.0), 1.0, 0.0);
    close(
        roex(P_FLOOR, 1000.0, 5000.0),
        1.4 * sva_formula::exp(-0.4),
        1e-15,
    );
}
