// only strings for now

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct Request {
    pub  query: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub  struct Response {
    pub  query: String,
    pub  result: String,
}