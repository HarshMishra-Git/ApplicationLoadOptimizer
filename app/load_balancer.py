import numpy as np
from collections import defaultdict

class LoadBalancer:
    def __init__(self, num_servers=3):
        self.num_servers = num_servers
        self.current_server = 0
        self.server_loads = defaultdict(list)
        self.server_response_times = defaultdict(list)
        
    def round_robin(self, request_load):
        """Implement round-robin load balancing"""
        selected_server = self.current_server
        
        # Simulate server load
        load = np.random.normal(request_load, request_load * 0.1)
        response_time = load * np.random.uniform(0.8, 1.2)
        
        self.server_loads[selected_server].append(load)
        self.server_response_times[selected_server].append(response_time)
        
        # Update current server
        self.current_server = (self.current_server + 1) % self.num_servers
        
        return selected_server, load, response_time
        
    def get_server_metrics(self):
        """Calculate server performance metrics"""
        metrics = {}
        
        for server in range(self.num_servers):
            metrics[f'server_{server}'] = {
                'avg_load': np.mean(self.server_loads[server]) if self.server_loads[server] else 0,
                'max_load': np.max(self.server_loads[server]) if self.server_loads[server] else 0,
                'avg_response_time': np.mean(self.server_response_times[server]) if self.server_response_times[server] else 0
            }
            
        return metrics
