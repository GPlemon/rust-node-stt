use hound;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use std::process::Command;
use std::path::Path;
use std::error::Error;
use std::fs;
use std::time::Instant;

fn fix_and_open_wav_inplace(path_str: &str) -> Result<hound::WavReader<std::io::BufReader<fs::File>>, Box<dyn Error>> {
    println!("Attempting to repair '{}' in-place with ffmpeg...", path_str);

    let input_path = Path::new(path_str);
    let temp_path = input_path.with_extension("repaired.tmp.wav");

    let output = Command::new("ffmpeg")
        .arg("-i")
        .arg(path_str)
        .arg("-c:a")
        .arg("copy")
        .arg("-y")
        .arg(&temp_path)
        .output()?;

    if !output.status.success() {
        let _ = fs::remove_file(&temp_path);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg failed to repair the file. Is ffmpeg installed and in your PATH?\nffmpeg stderr: {}", 
            stderr
        ).into());
    }

    fs::rename(&temp_path, path_str)?;
    println!("Successfully repaired and replaced '{}'.", path_str);

    hound::WavReader::open(path_str).map_err(|e| {
        format!("Failed to open the now-repaired file '{}': {}", path_str, e).into()
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let input_filename = "audio.wav";
    let mut reader = fix_and_open_wav_inplace(input_filename)?;
    
    let spec = reader.spec();
    println!("Sample rate: {}, Channels: {}, Bits per sample: {}", 
             spec.sample_rate, spec.channels, spec.bits_per_sample);
    
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Whisper works best with 16kHz audio. Current: {}Hz", spec.sample_rate);
    }
    
    let audio_data: Vec<f32> = match spec.bits_per_sample {
        16 => {
            if spec.channels == 2 {
                let samples = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;
                samples.chunks_exact(2).map(|chunk| {
                    let left = chunk[0] as f32 / 32768.0;
                    let right = chunk[1] as f32 / 32768.0;
                    (left + right) / 2.0
                }).collect()
            } else {
                reader.samples::<i16>()
                    .map(|s| s.map(|sample| sample as f32 / 32768.0))
                    .collect::<Result<Vec<f32>, _>>()?
            }
        },
        32 => {
            if spec.channels == 2 {
                let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
                samples.chunks_exact(2).map(|chunk| (chunk[0] + chunk[1]) / 2.0).collect()
            } else {
                reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
            }
        },
        _ => return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into()),
    };
    
    println!("Loaded {} audio samples", audio_data.len());
    
    let model_path = "models/ggml-base.en.bin";
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .expect("failed to load model");
    
    let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 1, patience: -1.0 });
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_timestamps(true);
    
    let mut state = ctx.create_state().expect("failed to create state");

    let start = Instant::now();
    state.full(params, &audio_data).expect("failed to run model");
    let duration = start.elapsed();
    println!("Transcription completed in {:.2?}", duration);
    
    println!("\nTranscription results:");
    for segment in state.as_iter() {
        println!("[{:.2}s - {:.2}s]: {}",
            segment.start_timestamp() as f64 / 1000.0,
            segment.end_timestamp() as f64 / 1000.0,
            segment
        );
    }
    
    Ok(())
}
