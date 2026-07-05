use std::collections::HashMap;
use std::rc::Rc;
use ndarray::Array1;

type StringStorage = HashMap<u64, Rc<String>>;
type VectorStorage = HashMap<u64, Array1<f64>>;

type ReverseStringStorage = HashMap<Rc<String>, u64>;
pub  struct InMemStore{
    string_storage: StringStorage,
    vector_storage: VectorStorage,
    present_vectors: ReverseStringStorage,
    index: u64
}

impl InMemStore {
    pub fn new() -> Self {
        InMemStore{ string_storage: HashMap::new(), vector_storage: HashMap::new(), present_vectors: HashMap::new(), index: 0 }
    }
    pub fn insert(self: &mut Self, data: &String, vector_data: &Array1<f64>) -> bool {
        if  self.present_vectors.contains_key(data){
            return false;
        }

        let text = Rc::new(String::from(data));

        self.string_storage.insert(self.index, Rc::clone(&text));
        self.vector_storage.insert(self.index, Array1::from_vec(vector_data.to_vec()));
        self.present_vectors.insert(Rc::clone(&text), self.index);
        self.index = self.index + 1;
        true
    }

    pub  fn get_store(&self) -> (&VectorStorage, &StringStorage){
        (&self.vector_storage, &self.string_storage)
    }

    pub fn get_vector_if_it_exists(&self, data: &String) -> Option<Array1<f64>> {
        // get index
        let index = self.present_vectors.get(data);
        match index {
            None => {
                None
            }
            Some(val) => {
                 Some(Array1::from(self.vector_storage.get(val).expect("Not Found").to_vec()))
            }
        }
    }

    pub fn get_index(&self, data: &String) -> u64 {
        // get index
        self.index
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test_insertion() {
        let mut mem_store = InMemStore::new();

        let input_text = String::from("abc");
        let input_embedding = vec![0.1, 0.2, 0.3];


        assert_eq!(mem_store.insert(&input_text, &Array1::from(input_embedding.clone())), true);
        assert_eq!(mem_store.insert(&input_text, &Array1::from(input_embedding)), false);
    }

    #[test]
    fn test_get_all_vectors() {
        let mut mem_store = InMemStore::new();

        let input_text = String::from("abc");
        let input_text2 = String::from("xyz");
        let input_embedding = vec![0.1, 0.2, 0.3];


        let _insert1 = mem_store.insert(&input_text, &Array1::from(input_embedding.clone()));
        let _insert2 = mem_store.insert(&input_text2, &Array1::from(input_embedding));

        let store = mem_store.get_store();

        assert_eq!(store.1.len(), 2);

    }

    #[test]
    fn test_get_vector() {
        let mut mem_store = InMemStore::new();

        let input_text = String::from("abc");
        let input_text2 = String::from("xyz");
        let input_embedding = vec![0.1, 0.2, 0.3];


        let _insert1 = mem_store.insert(&input_text, &Array1::from(input_embedding.clone()));
        let _insert2 = mem_store.insert(&input_text2, &Array1::from(input_embedding));

        let vector = mem_store.get_vector_if_it_exists(&input_text).expect("Should be found");

        assert_eq!(vector, Array1::from(vec![0.1, 0.2, 0.3]));

    }
}