from lepaute import main, load_data
import asyncio
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run the pipeline
asyncio.run(main())

# Access data
data = load_data()
for item in data:
    print(f"Lie params: {item['lie_params']}, Loss: {item['loss']}")
