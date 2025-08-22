use hound;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load WAV
    let mut reader = hound::WavReader::open("fixed_audio.wav")?; // Use the fixed file
    let spec = reader.spec();
    println!("Sample rate: {}, Channels: {}, Bits per sample: {}", 
             spec.sample_rate, spec.channels, spec.bits_per_sample);
    
    // Check if sample rate is 16kHz (Whisper expects this)
    if spec.sample_rate != 16000 {
        eprintln!("Warning: Whisper works best with 16kHz audio. Current: {}Hz", spec.sample_rate);
    }
    
    // Read samples based on the actual format (16-bit PCM)
    let audio_data: Vec<f32> = match spec.bits_per_sample {
        16 => {
            // Read as i16 and convert to f32 normalized to [-1.0, 1.0]
            if spec.channels == 2 {
                // Stereo to mono conversion
                let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
                let samples = samples?;
                let mut mono = Vec::with_capacity(samples.len() / 2);
                
                for chunk in samples.chunks_exact(2) {
                    let left = chunk[0] as f32 / 32768.0;
                    let right = chunk[1] as f32 / 32768.0;
                    mono.push((left + right) / 2.0);
                }
                mono
            } else {
                // Mono - convert i16 to f32
                reader.samples::<i16>()
                    .map(|s| s.map(|sample| sample as f32 / 32768.0))
                    .collect::<Result<Vec<f32>, _>>()?
            }
        },
        32 => {
            // Already f32 format
            if spec.channels == 2 {
                let samples: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
                let samples = samples?;
                let mut mono = Vec::with_capacity(samples.len() / 2);
                
                for chunk in samples.chunks_exact(2) {
                    mono.push((chunk[0] + chunk[1]) / 2.0);
                }
                mono
            } else {
                reader.samples::<f32>().collect::<Result<Vec<f32>, _>>()?
            }
        },
        _ => {
            return Err(format!("Unsupported bit depth: {}", spec.bits_per_sample).into());
        }
    };
    
    println!("Loaded {} audio samples", audio_data.len());
    
    // Load Whisper model
    let model_path = "modals/ggml-base.en.bin";
    let ctx = WhisperContext::new_with_params(
        model_path,
        WhisperContextParameters::default()
    ).expect("failed to load model");
    
    // Set transcription params
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: -1.0,
    });
    
    // Set language to English for better performance
    params.set_language(Some("en"));
    params.set_print_progress(true);
    params.set_print_timestamps(true);
    
    // Create state and transcribe
    let mut state = ctx.create_state().expect("failed to create state");
    state.full(params, &audio_data).expect("failed to run model");
    
    // Print results
    println!("\nTranscription results:");
    for segment in state.as_iter() {
        println!(
            "[{:.2}s - {:.2}s]: {}",
            segment.start_timestamp() as f64 / 1000.0,
            segment.end_timestamp() as f64 / 1000.0,
            segment
        );
    }
    
    Ok(())
}