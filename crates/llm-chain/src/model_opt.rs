use std::collections::HashMap;

use crate::tokens::PromptTokensError;
use serde::{Deserialize, Serialize};
use tiktoken_rs::{tokenizer::Tokenizer, CoreBPE};

const FALLBACK_CONTEXT_SIZE: i32 = 4096;

lazy_static::lazy_static! {
  static ref TOKENIZER: HashMap<Tokenizer, String> = {
    let mut map = HashMap::new();
    map.insert(Tokenizer::Cl100kBase, "gpt-4-1106-preview".to_string());
    map.insert(Tokenizer::P50kBase, "text-davinci-003".to_string());
    map.insert(Tokenizer::Gpt2, "gpt2".to_string());
    map
  };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOpt {
  id: String,
  name: String
}

impl ModelOpt {
  pub fn new(name: String) -> Self {
    Self { id: name.clone(), name }
  }

  pub fn new_with_bpe(name: String, tkn: Tokenizer) -> Self {
    Self { id: TOKENIZER.get(&tkn).unwrap().to_string(), name }
  }

  pub fn bpe(&self) -> Result<CoreBPE, PromptTokensError> {
    Ok(tiktoken_rs::get_bpe_from_model(&self.id()).unwrap())
  }

  pub fn id(&self) -> String {
    self.id.clone()
  }

  pub fn name(&self) -> String {
    self.name.clone()
  }

  pub fn max_tokens_allowed(&self) -> i32 {
    tiktoken_rs::model::get_context_size(&self.id()).try_into().unwrap_or(FALLBACK_CONTEXT_SIZE)
  }
}

impl From<ModelOpt> for Tokenizer {
  fn from(model: ModelOpt) -> Self {
    tiktoken_rs::tokenizer::get_tokenizer(&model.id()).unwrap()
  }
}
