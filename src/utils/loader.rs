use std::error::Error;

use csv::ReaderBuilder;
use reqwest::get;
use serde::de::DeserializeOwned;

pub async fn download_and_load_csv<T>(url: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: DeserializeOwned + Send,
{
    let response = get(url).await?;
    let content = response.text().await?;

    let mut csv_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(content.as_bytes());

    let inputs: Result<Vec<T>, _> = csv_reader
        .deserialize()
        .into_iter()
        .map(|input| input)
        .collect();

    Ok(inputs?)
}
