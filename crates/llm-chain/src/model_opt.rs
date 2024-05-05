use std::collections::HashMap;

use crate::tokens::PromptTokensError;
use serde::{Deserialize, Serialize};
use tiktoken_rs::{tokenizer::{Tokenizer}, CoreBPE};

lazy_static::lazy_static! {
  static ref TOKENIZER: HashMap<Tokenizer, String> = {
    let mut map = HashMap::new();
    map.insert(Tokenizer::Cl100kBase, "gpt-4-1106-preview".to_string());
    map.insert(Tokenizer::P50kBase, "text-davinci-003".to_string());
    map
  };
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOpt {
  Known(String),
  Unknown(String, String),
}

impl ModelOpt {
  pub fn new(name: String) -> Self {
    Self::Known(name)
  }

  pub fn new_with_bpe(name: String, tkn: Tokenizer) -> Self {
    Self::Unknown(name, TOKENIZER.get(&tkn).unwrap().to_string())
  }

  pub fn bpe(&self) -> Result<CoreBPE, PromptTokensError> {
    match self {
      Self::Known(name) => Ok(tiktoken_rs::get_bpe_from_model(name).unwrap()),
      Self::Unknown(_, fake) => Ok(tiktoken_rs::get_bpe_from_model(fake).unwrap()),
    }
  }

  pub fn id(&self) -> String {
    match self {
      Self::Known(name) => name.clone(),
      Self::Unknown(_, id) => id.clone(),
    }
  }

  pub fn name(&self) -> String {
    match self {
      Self::Known(name) => name.clone(),
      Self::Unknown(name, _) => name.clone(),
    }
  }
}

impl From<ModelOpt> for Tokenizer {
  fn from(model: ModelOpt) -> Self {
    tiktoken_rs::tokenizer::get_tokenizer(&model.id()).unwrap()
  }
}
