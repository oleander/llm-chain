use llm_chain::{executor, options, parameters, prompt, Parameters};
use llm_chain::traits::Executor as ExecutorTrait;
use llm_chain_openai::chatgpt::Executor;
use llm_chain::options::{ModelRef, Options};
use llm_chain::chains::map_reduce::Chain;
use serde::{Deserialize, Serialize};
use llm_chain::step::Step;
use anyhow::Result;
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new ChatGPT executor with the default settings
    let options = options!(
        Model: ModelRef::from_model_name("llama3:latest")
    );

    let exec = executor!(chatgpt, options)?;

    // Create the "map" step to summarize an article into bullet points
    let map_prompt = Step::for_prompt_template(prompt!(
        "You are a bot for summarizing wikipedia articles, you are terse and focus on accuracy",
        "Summarize this article into bullet points:\n{{text}}"
    ));

    // Create the "reduce" step to combine multiple summaries into one
    let reduce_prompt = Step::for_prompt_template(prompt!(
        "You are a diligent bot that summarizes text",
        "Please combine the articles below into one summary as bullet points:\n{{text}}"
    ));

    // Create a map-reduce chain with the map and reduce steps
    let chain = Chain::new(map_prompt, reduce_prompt);

    // Load the content of the article to be summarized
    let article = include_str!("article_to_summarize.md");

    // Create a vector with the Parameters object containing the text of the article
    let docs = vec![parameters!(article)];

    // Run the chain with the provided documents and an empty Parameters object for the "reduce" step
    let res = chain.run(docs, Parameters::new(), &exec).await?;

    // Print the result to the console
    println!("{}", res);
    Ok(())
}
