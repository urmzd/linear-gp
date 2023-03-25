use std::error::Error;

use csv::ReaderBuilder;
use reqwest::get;
use serde::de::DeserializeOwned;

pub async fn download_and_load_csv<T>(url: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: DeserializeOwned + Send,
{
    let response = get(url).await?.text().await?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(response.as_bytes());
    let mut data = Vec::new();

    for record in reader.deserialize() {
        let item: T = record?;
        data.push(item);
    }

    Ok(data)
}
