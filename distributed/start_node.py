import asyncio
import logging
import signal
import sys
from node_service import NodeService
import argparse
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        logger.info("Received shutdown signal...")
        self.kill_now = True

async def run_node(node_id: str, registry: str, port: int, killer: GracefulKiller):
    """Run the node service."""
    node = None
    try:
        # Initialize node service
        node = NodeService(node_id, registry, port)
        logger.info(f"Starting node {node_id}")
        
        # Run node until shutdown signal
        await node.run()
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if node:
            # Ensure node is stopped
            node.running = False
            await asyncio.sleep(0.5)  # Give time for cleanup
        logger.info("Node shutdown complete")

async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
        
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def main():
    parser = argparse.ArgumentParser(description='Start a training node')
    parser.add_argument('--node-id', required=True, help='Unique identifier for this node')
    parser.add_argument('--registry', required=True, help='Registry server address')
    parser.add_argument('--port', type=int, default=5555, help='Port to listen for jobs')
    parser.add_argument('--external-port', type=int, help='External port if behind NAT')
    
    args = parser.parse_args()
    
    # Setup shutdown handler
    killer = GracefulKiller()
    
    # Get event loop
    loop = asyncio.get_event_loop()
    
    # Add signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )
    
    try:
        loop.run_until_complete(
            run_node(args.node_id, args.registry, args.port, killer)
        )
    finally:
        loop.close()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main() 