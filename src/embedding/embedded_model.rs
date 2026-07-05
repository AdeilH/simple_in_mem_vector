use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

pub struct OnnxHelper {
    session: Session, // ort session
    model_location: String,
    tokenizer_location: String,
    tokenizer: Tokenizer,
}

impl OnnxHelper {
    pub fn new(model_location: String, tokenizer_location: String) -> Self {
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_inter_threads(8)
            .unwrap()
            .commit_from_file(&model_location)
            .unwrap();

        let tokenizer = Tokenizer::from_file(&tokenizer_location).unwrap();

        OnnxHelper {session, model_location, tokenizer_location, tokenizer}
    }

    pub fn embedded_model(&mut self, batched_inputs: &[&str], query: &str) {
        // insert query at the beginning
        let mut batched_vec = batched_inputs.to_vec();
        batched_vec.insert(0, query);
        let inputs = batched_vec.as_slice();
        let encodings = self.tokenizer.encode_batch(inputs.to_vec(), false).unwrap();

        let batch_size = encodings.len();
        let padded_token_length = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut ids: Vec<i64> = Vec::with_capacity(batch_size * padded_token_length);
        let mut mask: Vec<i64> = Vec::with_capacity(batch_size * padded_token_length);
        let mut type_ids: Vec<i64> = Vec::with_capacity(batch_size * padded_token_length);

        for encoding in &encodings {
            let current_ids = encoding.get_ids();
            let current_mask = encoding.get_attention_mask();
            let current_type_ids = encoding.get_type_ids();

            ids.extend(current_ids.iter().map(|&i| i as i64));
            mask.extend(current_mask.iter().map(|&i| i as i64));
            type_ids.extend(current_type_ids.iter().map(|&i| i as i64));

            let padding_needed = padded_token_length - encoding.len();
            if padding_needed > 0 {
                ids.resize(ids.len() + padding_needed, 0);
                mask.resize(mask.len() + padding_needed, 0);
                type_ids.resize(type_ids.len() + padding_needed, 0);
            }
        }

        let a_ids = TensorRef::from_array_view(([batch_size, padded_token_length], &*ids)).unwrap();
        let a_mask = TensorRef::from_array_view(([batch_size, padded_token_length], &*mask)).unwrap();
        let a_type_ids =
            TensorRef::from_array_view(([batch_size, padded_token_length], &*type_ids)).unwrap();

        let outputs = self.session
            .run(ort::inputs![a_ids, a_mask, a_type_ids])
            .unwrap();

        let raw_embeddings = outputs[0]
            .try_extract_array::<f32>()
            .unwrap()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();

        let embeddings = raw_embeddings.index_axis(ndarray::Axis(1), 0).to_owned();

        println!("Similarity for '{}'", inputs[0]);
        let query = embeddings.index_axis(ndarray::Axis(0), 0);

        let query_magnitude: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();

        for (emb, sentence) in embeddings
            .axis_iter(ndarray::Axis(0))
            .zip(inputs.iter())
            .skip(1)
        {
            let dot_product: f32 = query.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();

            let emb_magnitude: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();

            let cosine_similarity = if query_magnitude > 0.0 && emb_magnitude > 0.0 {
                dot_product / (query_magnitude * emb_magnitude)
            } else {
                0.0
            };

            println!("\t'{}': {:.1}%", sentence, cosine_similarity * 100.);
        }
    }
}

