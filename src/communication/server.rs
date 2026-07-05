use super::serialization::{Request, Response};


use actix_web::{get, web, App, HttpServer, Responder, test, HttpResponse, post};
use serde::Serialize;

#[get("/")]
pub  async fn greet(name: web::Path<String>) -> impl Responder {
    HttpResponse::Ok().body("Hello!".to_string())
}

#[post("/search")]
pub  async fn search(data: web::Json<Request>) -> impl  Responder {
    // receives requests
    let response: Response = Response{ query: data.into_inner().query, result: "".to_string()};
    HttpResponse::Ok().json(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[actix_web::test]
    async fn test_search() {
        let app = test::init_service(App::new().service(search)).await;

        // Build a mock POST request payload
        let payload = Request { query: "Alice".to_string() };

        let req = test::TestRequest::post()
            .uri("/search")
            .set_json(&payload)
            .to_request();

        // Send request and parse the JSON response body directly
        let response_body: Response = test::call_and_read_body_json(&app, req).await;

        assert_eq!(response_body.query, "Alice");
    }
}