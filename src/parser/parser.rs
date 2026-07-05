use pdfsink_rs::PdfDocument;
use std::collections::HashMap;
use text_splitter::TextSplitter;

#[derive(Debug, Clone)]
pub struct ChunkLocation {
    pub file_name: String,
    pub page_number: u16,
    pub paragraph_number: u16,
}

impl ChunkLocation {
    pub fn new() -> Self {
        ChunkLocation {
            file_name: "".to_string(),
            page_number: 0,
            paragraph_number: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataChunk {
    pub  chunk_content: String,
    pub  location: ChunkLocation,
}

impl DataChunk {
    pub fn new() -> Self {
        DataChunk {
            chunk_content: "".to_string(),
            location: ChunkLocation::new(),
        }
    }
}

pub  struct PDFParser {
    pub  data_chunk: DataChunk,
    pub  chunk_location: ChunkLocation,
}

impl PDFParser {
    pub fn new() -> Self {
        PDFParser {
            data_chunk: DataChunk::new(),
            chunk_location: ChunkLocation::new(),
        }
    }

    pub fn parse_file(&mut self, file_name: &str) -> HashMap<u16, String> {
        let pdf = PdfDocument::open(file_name);
        self.chunk_location.file_name = file_name.to_string();
        let mut pages_map = HashMap::new();
        match pdf {
            Ok(val) => {
                for page in val.pages {
                    let text = page.extract_text();
                    pages_map.insert(page.page_number as u16, text);
                }
            }
            Err(e) => {
                eprintln!("{}", e)
            }
        }
        pages_map
    }

    pub fn parse_pages_into_paragraphs(&mut self, file_name: &str) -> HashMap<(u16, u16), DataChunk> {
        let pages_map = self.parse_file(file_name);
        let mut result: HashMap<(u16, u16), DataChunk> = HashMap::new();
        for page in pages_map {
            self.chunk_location.page_number = page.0;
            // Maximum number of characters in a chunk
            let max_characters = 500;
            // Default implementation uses character count for chunk size
            let splitter = TextSplitter::new(max_characters);

            let chunks = splitter.chunks(page.1.as_str());

            for (num, chunk) in  chunks.into_iter().enumerate() {
                self.data_chunk.chunk_content = String::from(chunk);
                self.data_chunk.location.paragraph_number = num as u16;
                let key = (self.chunk_location.page_number , self.data_chunk.location.paragraph_number);
                result.insert(key, self.data_chunk.clone());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_extraction() {
        let mut pdf_parser = PDFParser::new();
        let parsed = pdf_parser.parse_file("sycl.pdf");
        assert_eq!(parsed.len(), 56);
    }
    #[test]
    fn test_chunking() {
        let mut pdf_parser = PDFParser::new();
        let parsed = pdf_parser.parse_pages_into_paragraphs("sycl.pdf");
    }
}
