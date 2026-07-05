use ndarray::Array1;
use openai_dive::v1::api::Client;
use openai_dive::v1::error::APIError;
use openai_dive::v1::resources::embedding::*;
pub struct EmbeddingClient {
    openai_client: Client,
    embedding_model: String,
}

impl EmbeddingClient {
    pub fn new(open_ai_key: String, open_ai_url: String, embedding_model: String) -> Self {
        let mut client = Client::new(open_ai_key);
        client.set_base_url(open_ai_url.as_str());
        EmbeddingClient {
            openai_client: client,
            embedding_model
        }
    }

    pub async  fn embed_string(self: &Self, input_text: &String) -> Option<Array1<f64>> {
        let builder = EmbeddingParametersBuilder::default()
        .model(self.embedding_model.clone())
        .input(EmbeddingInput::String(input_text.clone()))
        .encoding_format(EmbeddingEncodingFormat::Float)
        .build()
        .unwrap();

        let embedding_responses = self.openai_client.embeddings().create(builder).await.unwrap().data;

        match &embedding_responses[0].embedding{
            EmbeddingOutput::Float(val) => {return Some(Array1::from(val.clone()));}
            EmbeddingOutput::Base64(_) => {return None}
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
