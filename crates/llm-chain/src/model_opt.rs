use crate::tokens::PromptTokensError;
use serde::Serialize;
use tiktoken_rs::CoreBPE;

#[derive(Debug, Clone)]
pub enum ModelOpt {
  Known(String),
  Unknown(String, CoreBPE),
}

impl ModelOpt {
  pub fn new(name: String) -> Self {
    Self::Known(name)
  }

  pub fn new_with_bpe(name: String, bpe: CoreBPE) -> Self {
    Self::Unknown(name, bpe)
  }

  pub fn bpe(&self) -> Result<CoreBPE, PromptTokensError> {
    match self {
      Self::Known(name) => Ok(tiktoken_rs::get_bpe_from_model(name).unwrap()),
      Self::Unknown(_, bpe) => Ok(bpe.clone()),
    }
  }

  pub fn to_name(&self) -> String {
    match self {
      Self::Known(name) => name.clone(),
      Self::Unknown(name, _) => name.clone(),
    }
  }
}

impl Serialize for ModelOpt {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    match self {
      Self::Known(name) => name.serialize(serializer),
      Self::Unknown(name, _) => name.serialize(serializer),
    }
  }
}
