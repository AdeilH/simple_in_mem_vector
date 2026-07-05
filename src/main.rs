pub mod communication;
mod embedding;
pub mod parser;
pub mod query_manager;
pub mod storage_engine;

use embedding::openai_client::EmbeddingClient;
use storage_engine::in_memory::InMemStore;

use ndarray::prelude::*;

use parser::parser::PDFParser;

use embedding::embedded_model::OnnxHelper;

#[tokio::main]
async fn main() {
    let env = dotenvy::dotenv();

    match env {
        Ok(a) => {
            println!("{:?}", a);
        }
        Err(e) => panic!("{}", e),
    }

    // Onnx Basic Implementation
    let mut onnx = OnnxHelper::new("model/model.onnx".to_string(), "model/tokenizer.json".to_string());
    let mut inputs = vec![
        "Hello How are you",
        "Hello How are you please let me know",
        "Hello How are you now",
    ];
    onnx.embedded_model(inputs.as_slice(), "Hello How are you");

    let open_ai_key = std::env::var("DASHSCOPE_API_KEY").expect("Key must be set");
    let open_ai_url = std::env::var("DASHBOARD_URL").expect("You must give url");
    let model = std::env::var("MODEL").expect("No model specified");
    let local = std::env::var("IS_LOCAL").unwrap_or_default();
    let mut is_local: bool = true;
    if !local.is_empty() && local.to_ascii_lowercase() == "false" {
        is_local = false
    }
    let input_text_1 = "Hello How are you".to_string();
    let input_text_2 = "Hello How are you please let me know".to_string();
    let input_text_3 = "Hello How are you now".to_string();

    let client = EmbeddingClient::new(open_ai_key, open_ai_url, model, is_local);
    let resp = client.embed_string(&input_text_1).await;

    let mut in_mem_store = InMemStore::new();

    match resp {
        None => {}
        Some(vecaa) => {
            in_mem_store.insert(&input_text_1, &vecaa);
        }
    }

    let resp2 = client.embed_string(&input_text_2).await;

    match resp2 {
        None => {}
        Some(vecaa) => {
            in_mem_store.insert(&input_text_2, &vecaa);
        }
    }

    let mut parser = PDFParser::new();
    let chunks = parser.parse_pages_into_paragraphs("sycl.pdf");

    for chunk in chunks {
        let chunk_vec = client.embed_string(&chunk.1.chunk_content).await;
        match chunk_vec {
            None => {}
            Some(vecaa) => {
                in_mem_store.insert(&chunk.1.chunk_content, &vecaa);
            }
        }
    }

    let resp3 = client.embed_string(&input_text_3).await;

    let all_vecs = in_mem_store.get_store();

    match resp3 {
        None => {}
        Some(val) => {
            for (i, _all_vec) in all_vecs.0.iter().enumerate() {
                let curr_vector = &all_vecs.0[&(i as u64)];
                let curr_string = &all_vecs.1[&(i as u64)];
                let similarity = val.dot(curr_vector);
                if (similarity > 0.5) {
                    println!("{}", i);
                    println!("{:?}", similarity);
                    println!("{}", curr_string);
                }
            }
        }
    }
}
