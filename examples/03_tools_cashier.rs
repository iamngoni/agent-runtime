//! A declarative agent with its own tools that mutate shared state.
//!
//! The `Cashier` agent owns an `Arc<Mutex<…>>` cart; its tools capture clones
//! of it, so the agent definition stays self-contained (unit tool context).
//!
//! Run with: `OPENAI_API_KEY=sk-... cargo run --example 03_tools_cashier`

use std::sync::{Arc, Mutex};

use agent_runtime::{Agent, AgentProviderKind, JsonTool, Llm, ToolRegistry};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Clone, Default)]
struct Cart {
    items: Arc<Mutex<Vec<String>>>,
}

#[derive(Deserialize, JsonSchema)]
struct AddItem {
    name: String,
}

struct Cashier {
    cart: Cart,
}

impl Agent for Cashier {
    fn instructions(&self) -> String {
        "You are a cashier. Use add_to_cart for each item, then checkout and \
         report the item count."
            .into()
    }

    fn model(&self) -> String {
        "gpt-4o".into()
    }

    fn tools(&self) -> ToolRegistry<()> {
        let mut r = ToolRegistry::new();

        let cart = self.cart.clone();
        r.register(JsonTool::new(
            "add_to_cart",
            "Add a named item to the cart",
            move |_ctx: (), args: AddItem| {
                let cart = cart.clone();
                async move {
                    cart.items.lock().unwrap().push(args.name.clone());
                    Ok(json!({ "ok": true, "added": args.name }))
                }
            },
        ));

        let cart = self.cart.clone();
        r.register(JsonTool::new(
            "checkout",
            "Finalise the sale and return the item count",
            move |_ctx: (), _args: serde_json::Value| {
                let cart = cart.clone();
                async move { Ok(json!({ "items": cart.items.lock().unwrap().len() })) }
            },
        ));

        r
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let llm = Llm::builder()
        .provider(AgentProviderKind::OpenAi)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .build()?;

    let cashier = Cashier {
        cart: Cart::default(),
    };

    let reply = llm
        .run(&cashier, "Two lattes and an espresso, then check out.")
        .await?;
    println!("{reply}");
    Ok(())
}
