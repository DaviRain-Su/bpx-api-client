# Backpack Exchange API Examples

This repository demonstrates how to interact with Backpack Exchange’s APIs.

## Configuration

Set the following environment variables:

- `BASE_URL` (optional, default: `https://api.backpack.exchange`)
- `WS_URL` (optional, default: `wss://ws.backpack.exchange`)
- `SECRET` (required): Your API secret key.

## Setup Instructions

Follow these steps to configure and run the examples: Inside examples folder

1. **Create a `.env` File**:
   ```bash
   touch .env
   ```

2. **Add Environment Variables to `.env` File**:
    ```bash
    BPX_API_KEY=your_api_key_here
    SECRET=your_api_secret_here
    BASE_URL=https://api.backpack.exchange
    WS_URL=wss://ws.backpack.exchange
   ```

## Running the Examples

- To list available tasks, run: `just`
- To retrieve all the open orders, run: `just orders`
- To subscribe to the RFQ private stream, run: `just rfq`
