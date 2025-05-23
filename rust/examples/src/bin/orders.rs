use bpx_api_client::{BpxClient, BACKPACK_API_BASE_URL};
use std::env;

#[tokio::main]
async fn main() {
    let base_url = env::var("BASE_URL").unwrap_or_else(|_| BACKPACK_API_BASE_URL.to_string());
    println!("base_url: {}", base_url);
    let secret = env::var("SECRET").expect("Missing SECRET environment variable");
    //println!("secret: {}", secret);

    let client = BpxClient::init(base_url, &secret, None).expect("Failed to initialize Backpack API client");
    //println!("client = {:?}", client);
    match client.get_open_orders(Some("SOL_USDC")).await {
        Ok(orders) => println!("Open Orders: {:?}", orders),
        Err(err) => eprintln!("Error: {:?}", err),
    }
}
