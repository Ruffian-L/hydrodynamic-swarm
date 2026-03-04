//! Quick GGUF inspector: dump metadata + tensor names from a GGUF file.
//! Usage: cargo run --bin gguf_inspect -- data/Qwen3.5-4B-Q5_K_M.gguf

use std::io::BufReader;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: gguf_inspect <path.gguf>");

    let mut file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = candle_core::quantized::gguf_file::Content::read(&mut reader)?;

    println!("=== GGUF Metadata ===");
    let mut keys: Vec<_> = ct.metadata.keys().collect();
    keys.sort();
    for key in &keys {
        let val = &ct.metadata[*key];
        // Print compact repr
        let s = format!("{:?}", val);
        if s.len() > 120 {
            println!("  {} = {}...", key, &s[..120]);
        } else {
            println!("  {} = {}", key, s);
        }
    }

    println!("\n=== Tensor Info ({} tensors) ===", ct.tensor_infos.len());
    let mut tensor_names: Vec<_> = ct.tensor_infos.keys().collect();
    tensor_names.sort();
    for name in &tensor_names {
        let info = &ct.tensor_infos[*name];
        println!(
            "  {} | shape: {:?} | dtype: {:?}",
            name, info.shape, info.ggml_dtype
        );
    }

    // Print unique prefixes (layer structure)
    println!("\n=== Layer Prefixes ===");
    let mut prefixes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for name in &tensor_names {
        if let Some(dot_pos) = name.find('.') {
            if let Some(second_dot) = name[dot_pos + 1..].find('.') {
                let prefix = &name[..dot_pos + 1 + second_dot];
                prefixes.insert(prefix.to_string());
            } else {
                prefixes.insert(name.to_string());
            }
        } else {
            prefixes.insert(name.to_string());
        }
    }
    for p in &prefixes {
        println!("  {}", p);
    }
    // Dump values of norm weights to check convention (1+w vs plain w)
    let device = candle_core::Device::Cpu;
    println!("\n=== Norm Weight Values (Convention Check) ===");
    for name in &[
        "blk.0.ssm_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
    ] {
        if let Ok(t) = ct.tensor(&mut reader, name, &device) {
            if let Ok(deq) = t.dequantize(&device) {
                let vals: Vec<f32> = deq.to_vec1().unwrap_or_default();
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                println!(
                    "  {} [first 5]: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}] mean={:.4}",
                    name, vals[0], vals[1], vals[2], vals[3], vals[4], mean
                );
            }
        }
    }

    Ok(())
}
