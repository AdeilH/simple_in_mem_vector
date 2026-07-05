use super::super::embedding::openai_client::EmbeddingClient;
pub  struct QueryManager{
    embedding_client: EmbeddingClient
}

impl QueryManager {
    pub  fn new(embedding_client: EmbeddingClient) -> Self {
        QueryManager{embedding_client}
    }
}