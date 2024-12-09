import zmq
import zmq.asyncio
import asyncio
import json
import logging
from typing import Dict, Any
import base64
import pickle
from models import SimpleModel 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobDispatcher:
    def __init__(self, registry_address: str):
        self.ctx = zmq.asyncio.Context()
        
        # Connect to registry
        self.registry_socket = self.ctx.socket(zmq.REQ)
        if registry_address.startswith('tcp://'):
            registry_address = registry_address[6:]
        self.registry_socket.connect(f"tcp://{registry_address}")
        
    async def get_available_nodes(self) -> Dict[str, Any]:
        """Query registry for available nodes."""
        try:
            logger.info("Querying registry for available nodes...")
            await self.registry_socket.send_json({'query': 'nodes'})
            response = await self.registry_socket.recv_json()
            nodes = response.get('nodes', {})
            logger.info(f"Registry response: {response}")
            return nodes
        except Exception as e:
            logger.error(f"Error querying registry: {e}")
            return {}
        
    async def dispatch_job(self, node_address: str, job_config: Dict):
        """Send job to a specific node."""
        # Connect to node
        node_socket = self.ctx.socket(zmq.REQ)
        node_socket.connect(node_address)
        
        try:
            # Prepare job config (convert model class to base64 encoded string)
            serializable_config = {
                'model_class': base64.b64encode(pickle.dumps(job_config['model_class'])).decode('utf-8'),
                'model_args': job_config['model_args'],
                'training_args': job_config['training_args']
            }
            
            # Send job configuration
            await node_socket.send_json({
                'command': 'start_job',
                'config': serializable_config
            })
            
            response = await node_socket.recv_json()
            logger.info(f"Node response: {response}")
            return response
            
        finally:
            node_socket.close()
            
    async def run(self, job_config: Dict):
        """Find available nodes and dispatch job."""
        try:
            # Get available nodes
            nodes = await self.get_available_nodes()
            if not nodes:
                logger.error("No available nodes found")
                return
                
            logger.info(f"Found {len(nodes)} available nodes")
            
            # Dispatch to each idle node
            for node_id, info in nodes.items():
                if info['status'] == 'idle':
                    logger.info(f"Dispatching job to node {node_id}")
                    await self.dispatch_job(info['address'], job_config)
                    
        finally:
            self.registry_socket.close()
            self.ctx.term()

async def main():
    # Example job configuration
    job_config = {
        'model_class': SimpleModel,
        'model_args': {
            'input_size': 784,
            'hidden_size': 128,
            'output_size': 10
        },
        'training_args': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10
        }
    }
    
    dispatcher = JobDispatcher("localhost:5556")
    await dispatcher.run(job_config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Dispatch training jobs to nodes')
    parser.add_argument('--registry', default='localhost:5556', help='Registry server address (e.g., localhost:5556)')
    
    args = parser.parse_args()
    
    asyncio.run(main()) 