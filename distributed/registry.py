import zmq
import zmq.asyncio
import asyncio
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegistryServer:
    def __init__(self, port=5556):
        self.port = port
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Track nodes and servers
        self.nodes = {}  # {node_id: {address, status, last_heartbeat}}
        self.training_servers = {}  # {server_id: {address, jobs}}
        self.running = True
        
    def _serialize_node_info(self, info):
        """Convert node info to JSON-serializable format."""
        serialized = {
            'address': info['address'],
            'status': info['status'],
            'device': info['device'],
            'last_heartbeat': info['last_heartbeat'].isoformat()
        }
        # Include jobs if present
        if 'jobs' in info:
            serialized['jobs'] = info['jobs']
        return serialized
        
    def _store_node_info(self, entity_id, address, status='idle', device='cpu', jobs=None):
        """Store node info with current timestamp."""
        info = {
            'address': address,
            'status': status,
            'device': device,
            'last_heartbeat': datetime.now()
        }
        if jobs is not None:
            info['jobs'] = jobs
        return info
        
    async def handle_heartbeat(self, msg):
        """Handle node/server heartbeats."""
        entity_type = msg['type']
        entity_id = msg['id']
        address = msg['address']
        
        if entity_type == 'node':
            if entity_id not in self.nodes:
                logger.info(f"New node registered: {entity_id} at {address}")
            self.nodes[entity_id] = self._store_node_info(
                entity_id,
                address,
                status=msg.get('status', 'idle'),
                device=msg.get('device', 'cpu')
            )
        elif entity_type == 'server':
            if entity_id not in self.training_servers:
                logger.info(f"New training server registered: {entity_id} at {address}")
            self.training_servers[entity_id] = self._store_node_info(
                entity_id,
                address,
                jobs=msg.get('jobs', [])
            )
            
        # Return serialized status
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        }
        
    async def handle_query(self, msg):
        """Handle queries for available nodes/servers."""
        query_type = msg['query']
        
        if query_type == 'nodes':
            # Return active nodes
            active_nodes = {
                node_id: self._serialize_node_info(info)
                for node_id, info in self.nodes.items()
                if datetime.now() - info['last_heartbeat'] < timedelta(seconds=30)
            }
            logger.info(f"Active nodes query - Found {len(active_nodes)} nodes: {active_nodes}")
            return {'nodes': active_nodes}
            
        elif query_type == 'servers':
            # Return active training servers
            active_servers = {
                server_id: self._serialize_node_info(info)
                for server_id, info in self.training_servers.items()
                if datetime.now() - info['last_heartbeat'] < timedelta(seconds=30)
            }
            return {'servers': active_servers}
            
    async def serve(self):
        """Main registry loop."""
        logger.info(f"Registry server running on port {self.port}")
        
        while self.running:
            try:
                msg = await self.socket.recv_json()
                
                if msg.get('type') in ['node', 'server']:
                    response = await self.handle_heartbeat(msg)
                elif msg.get('query'):
                    response = await self.handle_query(msg)
                else:
                    response = {'error': 'invalid message'}
                    
                await self.socket.send_json(response)
                
            except Exception as e:
                logger.error(f"Error in registry server: {e}")
                await self.socket.send_json({'error': str(e)})
                
    def start(self):
        """Start the registry server."""
        asyncio.run(self.serve())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5556, help="Registry port")
    args = parser.parse_args()
    
    registry = RegistryServer(args.port)
    registry.start() 