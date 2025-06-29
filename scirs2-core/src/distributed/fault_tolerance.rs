//! Fault tolerance mechanisms for distributed systems
//!
//! This module provides fault detection, recovery, and resilience mechanisms
//! for distributed computing environments.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Node health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeHealth {
    /// Node is healthy and responsive
    Healthy,
    /// Node is degraded but functional
    Degraded,
    /// Node is unresponsive
    Unresponsive,
    /// Node has failed
    Failed,
    /// Node is recovering
    Recovering,
}

/// Fault detection strategy
#[derive(Debug, Clone)]
pub enum FaultDetectionStrategy {
    /// Heartbeat-based detection
    Heartbeat {
        interval: Duration,
        timeout: Duration,
    },
    /// Ping-based detection
    Ping {
        interval: Duration,
        timeout: Duration,
    },
    /// Application-level health checks
    HealthCheck {
        interval: Duration,
        endpoint: String,
    },
}

/// Recovery strategy for failed nodes
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Restart the node
    Restart,
    /// Migrate tasks to healthy nodes
    Migrate,
    /// Replace with standby node
    Replace { standby_address: SocketAddr },
    /// Manual intervention required
    Manual,
}

/// Node information for fault tolerance
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: SocketAddr,
    pub health: NodeHealth,
    pub last_seen: Instant,
    pub failure_count: usize,
    pub recovery_strategy: RecoveryStrategy,
}

impl NodeInfo {
    /// Create new node info
    pub fn new(node_id: String, address: SocketAddr) -> Self {
        Self {
            node_id,
            address,
            health: NodeHealth::Healthy,
            last_seen: Instant::now(),
            failure_count: 0,
            recovery_strategy: RecoveryStrategy::Restart,
        }
    }

    /// Update node health status
    pub fn update_health(&mut self, health: NodeHealth) {
        if health == NodeHealth::Failed {
            self.failure_count += 1;
        }
        self.health = health;
        self.last_seen = Instant::now();
    }

    /// Check if node is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.health, NodeHealth::Healthy)
    }

    /// Check if node has failed
    pub fn has_failed(&self) -> bool {
        matches!(self.health, NodeHealth::Failed | NodeHealth::Unresponsive)
    }
}

/// Fault tolerance manager
#[derive(Debug)]
pub struct FaultToleranceManager {
    nodes: Arc<Mutex<HashMap<String, NodeInfo>>>,
    detection_strategy: FaultDetectionStrategy,
    max_failures: usize,
    failure_threshold: Duration,
}

impl FaultToleranceManager {
    /// Create a new fault tolerance manager
    pub fn new(detection_strategy: FaultDetectionStrategy, max_failures: usize) -> Self {
        Self {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            detection_strategy,
            max_failures,
            failure_threshold: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Register a node for monitoring
    pub fn register_node(&self, node_info: NodeInfo) -> CoreResult<()> {
        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;
        nodes.insert(node_info.node_id.clone(), node_info);
        Ok(())
    }

    /// Update node health status
    pub fn update_node_health(&self, node_id: &str, health: NodeHealth) -> CoreResult<()> {
        let mut nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        if let Some(node) = nodes.get_mut(node_id) {
            node.update_health(health);
        } else {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Unknown node: {}",
                node_id
            ))));
        }
        Ok(())
    }

    /// Get all healthy nodes
    pub fn get_healthy_nodes(&self) -> CoreResult<Vec<NodeInfo>> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        Ok(nodes
            .values()
            .filter(|node| node.is_healthy())
            .cloned()
            .collect())
    }

    /// Get all failed nodes
    pub fn get_failed_nodes(&self) -> CoreResult<Vec<NodeInfo>> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        Ok(nodes
            .values()
            .filter(|node| node.has_failed())
            .cloned()
            .collect())
    }

    /// Detect failed nodes based on timeout
    pub fn detect_failures(&self) -> CoreResult<Vec<String>> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let now = Instant::now();
        let mut failed_nodes = Vec::new();

        for (node_id, node) in nodes.iter() {
            let timeout = match &self.detection_strategy {
                FaultDetectionStrategy::Heartbeat { timeout, .. } => *timeout,
                FaultDetectionStrategy::Ping { timeout, .. } => *timeout,
                FaultDetectionStrategy::HealthCheck { .. } => Duration::from_secs(30),
            };

            if now.duration_since(node.last_seen) > timeout && node.is_healthy() {
                failed_nodes.push(node_id.clone());
            }
        }

        Ok(failed_nodes)
    }

    /// Initiate recovery for failed nodes
    pub fn initiate_recovery(&self, node_id: &str) -> CoreResult<()> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        if let Some(node) = nodes.get(node_id) {
            match &node.recovery_strategy {
                RecoveryStrategy::Restart => {
                    self.restart_node(node_id)?;
                }
                RecoveryStrategy::Migrate => {
                    self.migrate_tasks(node_id)?;
                }
                RecoveryStrategy::Replace { standby_address } => {
                    self.replace_node(node_id, *standby_address)?;
                }
                RecoveryStrategy::Manual => {
                    println!("Manual intervention required for node: {}", node_id);
                }
            }
        }

        Ok(())
    }

    fn restart_node(&self, node_id: &str) -> CoreResult<()> {
        // In a real implementation, this would trigger node restart
        println!("Restarting node: {}", node_id);
        Ok(())
    }

    fn migrate_tasks(&self, node_id: &str) -> CoreResult<()> {
        // In a real implementation, this would migrate tasks to healthy nodes
        println!("Migrating tasks from failed node: {}", node_id);
        Ok(())
    }

    fn replace_node(&self, node_id: &str, standby_address: SocketAddr) -> CoreResult<()> {
        // In a real implementation, this would activate standby node
        println!(
            "Replacing node {} with standby at {}",
            node_id, standby_address
        );
        Ok(())
    }

    /// Check if the cluster has sufficient healthy nodes
    pub fn is_cluster_healthy(&self) -> CoreResult<bool> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let healthy_count = nodes.values().filter(|node| node.is_healthy()).count();
        let total_count = nodes.len();

        // Require at least 50% of nodes to be healthy
        Ok(healthy_count * 2 >= total_count)
    }

    /// Get cluster health summary
    pub fn get_cluster_health_summary(&self) -> CoreResult<ClusterHealthSummary> {
        let nodes = self.nodes.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire nodes lock".to_string(),
            ))
        })?;

        let mut summary = ClusterHealthSummary::default();

        for node in nodes.values() {
            match node.health {
                NodeHealth::Healthy => summary.healthy_count += 1,
                NodeHealth::Degraded => summary.degraded_count += 1,
                NodeHealth::Unresponsive => summary.unresponsive_count += 1,
                NodeHealth::Failed => summary.failed_count += 1,
                NodeHealth::Recovering => summary.recovering_count += 1,
            }
        }

        summary.total_count = nodes.len();
        Ok(summary)
    }
}

/// Cluster health summary
#[derive(Debug, Default)]
pub struct ClusterHealthSummary {
    pub total_count: usize,
    pub healthy_count: usize,
    pub degraded_count: usize,
    pub unresponsive_count: usize,
    pub failed_count: usize,
    pub recovering_count: usize,
}

impl ClusterHealthSummary {
    /// Calculate health percentage
    pub fn health_percentage(&self) -> f64 {
        if self.total_count == 0 {
            return 100.0;
        }
        (self.healthy_count as f64 / self.total_count as f64) * 100.0
    }

    /// Check if cluster is in good health
    pub fn is_healthy(&self) -> bool {
        self.health_percentage() >= 75.0
    }
}

/// Initialize fault tolerance system
pub fn initialize_fault_tolerance() -> CoreResult<()> {
    let _manager = FaultToleranceManager::new(
        FaultDetectionStrategy::Heartbeat {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
        },
        3,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_node_info_creation() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new("node1".to_string(), address);

        assert_eq!(node.node_id, "node1");
        assert_eq!(node.address, address);
        assert_eq!(node.health, NodeHealth::Healthy);
        assert!(node.is_healthy());
        assert!(!node.has_failed());
    }

    #[test]
    fn test_node_health_update() {
        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let mut node = NodeInfo::new("node1".to_string(), address);

        node.update_health(NodeHealth::Failed);
        assert_eq!(node.health, NodeHealth::Failed);
        assert_eq!(node.failure_count, 1);
        assert!(node.has_failed());
    }

    #[test]
    fn test_fault_tolerance_manager() {
        let strategy = FaultDetectionStrategy::Heartbeat {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
        };
        let manager = FaultToleranceManager::new(strategy, 3);

        let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);
        let node = NodeInfo::new("node1".to_string(), address);

        assert!(manager.register_node(node).is_ok());
        assert!(manager
            .update_node_health("node1", NodeHealth::Failed)
            .is_ok());

        let failed_nodes = manager.get_failed_nodes().unwrap();
        assert_eq!(failed_nodes.len(), 1);
        assert_eq!(failed_nodes[0].node_id, "node1");
    }

    #[test]
    fn test_cluster_health_summary() {
        let summary = ClusterHealthSummary {
            total_count: 10,
            healthy_count: 8,
            failed_count: 2,
            ..Default::default()
        };

        assert_eq!(summary.health_percentage(), 80.0);
        assert!(summary.is_healthy());
    }
}
