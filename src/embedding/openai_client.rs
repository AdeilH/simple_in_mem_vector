use ndarray::Array1;
use openai_dive::v1::api::Client;
use openai_dive::v1::error::APIError;
use openai_dive::v1::resources::embedding::*;
use serde::{Deserialize, Serialize};

pub struct EmbeddingClient {
    openai_client: Client,
    embedding_model: String,
    open_ai_url: String,
    is_local: bool,
}

#[derive(Serialize, Deserialize)]
pub struct LocalRequest {
    input: String,
    model: String,
}

// 1. Manually deserialize into a custom flexible struct that doesn't care about missing string flags
#[derive(serde::Deserialize, Debug)]
pub struct LlamaResponse {
    pub data: Vec<LlamaItem>,
}

#[derive(serde::Deserialize, Debug)]
pub struct LlamaItem {
    pub embedding: Vec<f64>, // This handles the float map safely
}

impl EmbeddingClient {
    pub fn new(open_ai_key: String, open_ai_url: String, embedding_model: String, is_local: bool) -> Self {
        let mut client = Client::new(open_ai_key);
        client.set_base_url(open_ai_url.as_str());
        EmbeddingClient {
            openai_client: client,
            embedding_model,
            open_ai_url,
            is_local,
        }
    }

    pub async fn embed_string(self: &Self, input_text: &String) -> Option<Array1<f64>> {
        if (self.is_local) {
            let http_client = reqwest::Client::new();
            let payload = LocalRequest {
                input: input_text.clone(),
                model: self.embedding_model.clone(),
            };

            let response = http_client
                .post(format!("{}/v1/embeddings", self.open_ai_url))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let parsed: LlamaResponse = response
                .json()
                .await
                .map_err(|e| {
                    println!("Extraction Error: {:?}", e);
                    e
                })
                .unwrap();

            if let Some(item) = parsed.data.first() {
                let array_from_vector = Array1::from(item.embedding.clone());
                return Some(Array1::from(array_from_vector));
            }
        }

        let builder = EmbeddingParametersBuilder::default()
            .model(self.embedding_model.clone())
            .input(EmbeddingInput::String(input_text.clone()))
            .encoding_format(EmbeddingEncodingFormat::Float)
            .build()
            .unwrap();

        let embedding_responses = self
            .openai_client
            .embeddings()
            .create(builder)
            .await
            .unwrap()
            .data;

        match &embedding_responses[0].embedding {
            EmbeddingOutput::Float(val) => {
                return Some(Array1::from(val.clone()));
            }
            EmbeddingOutput::Base64(_) => return None,
        }
    }
}

// let open_ai_key = std::env::var("DASHSCOPE_API_KEY").expect("Key must be set");
// let open_ai_url = std::env::var("DASHBOARD_URL").expect("You must give url");
// let model = std::env::var("MODEL").expect("No model specified");
//
// let mut client = Client::new(open_ai_key);
// client.set_base_url(&open_ai_url);
//
// let builder = EmbeddingParametersBuilder::default()
// .model(model)
// .input(EmbeddingInput::String("Hello".to_string()))
// .encoding_format(EmbeddingEncodingFormat::Float)
// .build()
// .unwrap();
//
// let result = client.embeddings().create(builder).await.unwrap();
//
// for r in result.data {
// let output = r.embedding;
// match output {
// EmbeddingOutput::Float(a) => {
// for d in a {
// println!("{}",d);
// }
// }
// EmbeddingOutput::Base64(b) => {}
// }
// }
