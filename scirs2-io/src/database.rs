//! Database connectivity for scientific data
//!
//! Provides interfaces for reading and writing scientific data to various
//! database systems, including SQL and NoSQL databases.

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Supported database types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatabaseType {
    /// PostgreSQL database
    PostgreSQL,
    /// MySQL/MariaDB database
    MySQL,
    /// SQLite database
    SQLite,
    /// MongoDB (NoSQL)
    MongoDB,
    /// InfluxDB (Time series)
    InfluxDB,
    /// Redis (Key-value)
    Redis,
    /// Cassandra (Wide column)
    Cassandra,
    /// DuckDB (Analytical)
    DuckDB,
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub db_type: DatabaseType,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub options: HashMap<String, String>,
}

impl DatabaseConfig {
    /// Create a new database configuration
    pub fn new(db_type: DatabaseType, database: impl Into<String>) -> Self {
        Self {
            db_type,
            host: None,
            port: None,
            database: database.into(),
            username: None,
            password: None,
            options: HashMap::new(),
        }
    }

    /// Set host
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set port
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set credentials
    pub fn credentials(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }

    /// Add connection option
    pub fn option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Build connection string
    pub fn connection_string(&self) -> String {
        match self.db_type {
            DatabaseType::PostgreSQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(5432);
                let user = self.username.as_deref().unwrap_or("postgres");
                format!("postgresql://{}:password@{}:{}/{}", user, host, port, self.database)
            }
            DatabaseType::MySQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(3306);
                let user = self.username.as_deref().unwrap_or("root");
                format!("mysql://{}:password@{}:{}/{}", user, host, port, self.database)
            }
            DatabaseType::SQLite => {
                format!("sqlite://{}", self.database)
            }
            DatabaseType::MongoDB => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(27017);
                format!("mongodb://{}:{}/{}", host, port, self.database)
            }
            _ => format!("{}://{}", self.db_type.as_str(), self.database),
        }
    }
}

impl DatabaseType {
    fn as_str(&self) -> &'static str {
        match self {
            Self::PostgreSQL => "postgresql",
            Self::MySQL => "mysql",
            Self::SQLite => "sqlite",
            Self::MongoDB => "mongodb",
            Self::InfluxDB => "influxdb",
            Self::Redis => "redis",
            Self::Cassandra => "cassandra",
            Self::DuckDB => "duckdb",
        }
    }
}

/// Database query builder
pub struct QueryBuilder {
    query_type: QueryType,
    table: String,
    columns: Vec<String>,
    conditions: Vec<String>,
    values: Vec<serde_json::Value>,
    order_by: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Debug, Clone)]
enum QueryType {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
}

impl QueryBuilder {
    /// Create a SELECT query
    pub fn select(table: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Select,
            table: table.into(),
            columns: vec!["*".to_string()],
            conditions: Vec::new(),
            values: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Create an INSERT query
    pub fn insert(table: impl Into<String>) -> Self {
        Self {
            query_type: QueryType::Insert,
            table: table.into(),
            columns: Vec::new(),
            conditions: Vec::new(),
            values: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
        }
    }

    /// Specify columns
    pub fn columns(mut self, columns: Vec<impl Into<String>>) -> Self {
        self.columns = columns.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Add WHERE condition
    pub fn where_clause(mut self, condition: impl Into<String>) -> Self {
        self.conditions.push(condition.into());
        self
    }

    /// Add values for INSERT
    pub fn values(mut self, values: Vec<serde_json::Value>) -> Self {
        self.values = values;
        self
    }

    /// Set ORDER BY
    pub fn order_by(mut self, column: impl Into<String>, desc: bool) -> Self {
        self.order_by = Some(format!("{} {}", column.into(), if desc { "DESC" } else { "ASC" }));
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build SQL query string
    pub fn build_sql(&self) -> String {
        match self.query_type {
            QueryType::Select => {
                let mut sql = format!("SELECT {} FROM {}", self.columns.join(", "), self.table);
                
                if !self.conditions.is_empty() {
                    sql.push_str(&format!(" WHERE {}", self.conditions.join(" AND ")));
                }
                
                if let Some(order) = &self.order_by {
                    sql.push_str(&format!(" ORDER BY {}", order));
                }
                
                if let Some(limit) = self.limit {
                    sql.push_str(&format!(" LIMIT {}", limit));
                }
                
                if let Some(offset) = self.offset {
                    sql.push_str(&format!(" OFFSET {}", offset));
                }
                
                sql
            }
            QueryType::Insert => {
                format!(
                    "INSERT INTO {} ({}) VALUES ({})",
                    self.table,
                    self.columns.join(", "),
                    self.values.iter().map(|_| "?").collect::<Vec<_>>().join(", ")
                )
            }
            _ => String::new(),
        }
    }

    /// Build MongoDB query
    pub fn build_mongo(&self) -> serde_json::Value {
        match self.query_type {
            QueryType::Select => {
                let mut query = serde_json::json!({});
                
                // Convert SQL-like conditions to MongoDB query
                for condition in &self.conditions {
                    // Simple parsing - in real implementation would be more sophisticated
                    if let Some((field, value)) = condition.split_once(" = ") {
                        query[field] = serde_json::json!(value.trim_matches('\''));
                    }
                }
                
                serde_json::json!({
                    "collection": self.table,
                    "filter": query,
                    "limit": self.limit,
                    "skip": self.offset,
                })
            }
            _ => serde_json::json!({}),
        }
    }
}

/// Database result set
#[derive(Debug, Clone)]
pub struct ResultSet {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub metadata: Metadata,
}

impl ResultSet {
    /// Create new result set
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            metadata: Metadata::new(),
        }
    }

    /// Add a row
    pub fn add_row(&mut self, row: Vec<serde_json::Value>) {
        self.rows.push(row);
    }

    /// Get number of rows
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Convert to Array2<f64> if all values are numeric
    pub fn to_array(&self) -> Result<Array2<f64>> {
        let mut data = Vec::new();
        
        for row in &self.rows {
            for value in row {
                let num = value.as_f64()
                    .ok_or_else(|| IoError::ConversionError("Non-numeric value in result set".to_string()))?;
                data.push(num);
            }
        }
        
        Array2::from_shape_vec((self.row_count(), self.column_count()), data)
            .map_err(|e| IoError::Other(e.to_string()))
    }

    /// Get column by name as Array1
    pub fn get_column(&self, name: &str) -> Result<Array1<f64>> {
        let col_idx = self.columns.iter().position(|c| c == name)
            .ok_or_else(|| IoError::Other(format!("Column '{}' not found", name)))?;
        
        let mut data = Vec::new();
        for row in &self.rows {
            let num = row[col_idx].as_f64()
                .ok_or_else(|| IoError::ConversionError("Non-numeric value in column".to_string()))?;
            data.push(num);
        }
        
        Ok(Array1::from_vec(data))
    }
}

/// Database connection trait
pub trait DatabaseConnection: Send + Sync {
    /// Execute a query and return results
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet>;
    
    /// Execute a raw SQL query
    fn execute_sql(&self, sql: &str, params: &[serde_json::Value]) -> Result<ResultSet>;
    
    /// Insert data from Array2
    fn insert_array(&self, table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize>;
    
    /// Create table from schema
    fn create_table(&self, table: &str, schema: &TableSchema) -> Result<()>;
    
    /// Check if table exists
    fn table_exists(&self, table: &str) -> Result<bool>;
    
    /// Get table schema
    fn get_schema(&self, table: &str) -> Result<TableSchema>;
}

/// Table schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<Vec<String>>,
    pub indexes: Vec<Index>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Integer,
    BigInt,
    Float,
    Double,
    Decimal(u8, u8),
    Varchar(usize),
    Text,
    Boolean,
    Date,
    Timestamp,
    Json,
    Binary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub name: String,
    pub columns: Vec<String>,
    pub unique: bool,
}

/// Database connector factory
pub struct DatabaseConnector;

impl DatabaseConnector {
    /// Create a new database connection
    pub fn connect(config: &DatabaseConfig) -> Result<Box<dyn DatabaseConnection>> {
        match config.db_type {
            DatabaseType::SQLite => Ok(Box::new(SQLiteConnection::new(config)?)),
            _ => Err(IoError::UnsupportedFormat(
                format!("Database type {:?} not yet implemented", config.db_type)
            )),
        }
    }
}

/// SQLite connection implementation
struct SQLiteConnection {
    path: String,
}

impl SQLiteConnection {
    fn new(config: &DatabaseConfig) -> Result<Self> {
        Ok(Self {
            path: config.database.clone(),
        })
    }
}

impl DatabaseConnection for SQLiteConnection {
    fn query(&self, query: &QueryBuilder) -> Result<ResultSet> {
        // Simplified implementation - would use actual SQLite library
        let mut result = ResultSet::new(query.columns.clone());
        
        // Mock some data
        if query.table == "test_table" {
            result.add_row(vec![
                serde_json::json!(1),
                serde_json::json!("test"),
                serde_json::json!(3.14),
            ]);
        }
        
        Ok(result)
    }

    fn execute_sql(&self, _sql: &str, _params: &[serde_json::Value]) -> Result<ResultSet> {
        Ok(ResultSet::new(vec![]))
    }

    fn insert_array(&self, _table: &str, data: ArrayView2<f64>, columns: &[&str]) -> Result<usize> {
        Ok(data.nrows())
    }

    fn create_table(&self, _table: &str, _schema: &TableSchema) -> Result<()> {
        Ok(())
    }

    fn table_exists(&self, _table: &str) -> Result<bool> {
        Ok(false)
    }

    fn get_schema(&self, table: &str) -> Result<TableSchema> {
        Ok(TableSchema {
            name: table.to_string(),
            columns: vec![],
            primary_key: None,
            indexes: vec![],
        })
    }
}

/// Time series database utilities
pub mod timeseries {
    use super::*;
    use chrono::{DateTime, Utc};
    
    /// Time series data point
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TimeSeriesPoint {
        pub timestamp: DateTime<Utc>,
        pub value: f64,
        pub tags: HashMap<String, String>,
    }
    
    /// Time series query builder
    pub struct TimeSeriesQuery {
        pub measurement: String,
        pub start_time: Option<DateTime<Utc>>,
        pub end_time: Option<DateTime<Utc>>,
        pub tags: HashMap<String, String>,
        pub aggregation: Option<Aggregation>,
        pub group_by: Vec<String>,
    }
    
    #[derive(Debug, Clone)]
    pub enum Aggregation {
        Mean,
        Sum,
        Min,
        Max,
        Count,
        StdDev,
    }
    
    impl TimeSeriesQuery {
        pub fn new(measurement: impl Into<String>) -> Self {
            Self {
                measurement: measurement.into(),
                start_time: None,
                end_time: None,
                tags: HashMap::new(),
                aggregation: None,
                group_by: Vec::new(),
            }
        }
        
        pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
            self.start_time = Some(start);
            self.end_time = Some(end);
            self
        }
        
        pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
            self.tags.insert(key.into(), value.into());
            self
        }
        
        pub fn aggregate(mut self, agg: Aggregation) -> Self {
            self.aggregation = Some(agg);
            self
        }
    }
}

/// Bulk operations for efficient data loading
pub mod bulk {
    use super::*;
    
    /// Bulk loader for efficient data insertion
    pub struct BulkLoader {
        batch_size: usize,
        buffer: Vec<Vec<serde_json::Value>>,
        table: String,
        columns: Vec<String>,
    }
    
    impl BulkLoader {
        pub fn new(table: impl Into<String>, columns: Vec<impl Into<String>>) -> Self {
            Self {
                batch_size: 1000,
                buffer: Vec::new(),
                table: table.into(),
                columns: columns.into_iter().map(|c| c.into()).collect(),
            }
        }
        
        pub fn batch_size(mut self, size: usize) -> Self {
            self.batch_size = size;
            self
        }
        
        pub fn add_row(&mut self, row: Vec<serde_json::Value>) -> Result<()> {
            if row.len() != self.columns.len() {
                return Err(IoError::Other("Row length mismatch".to_string()));
            }
            self.buffer.push(row);
            Ok(())
        }
        
        pub fn add_array(&mut self, data: ArrayView2<f64>) -> Result<()> {
            for row in data.rows() {
                let json_row: Vec<serde_json::Value> = row.iter()
                    .map(|&v| serde_json::json!(v))
                    .collect();
                self.add_row(json_row)?;
            }
            Ok(())
        }
        
        pub fn flush(&mut self, conn: &dyn DatabaseConnection) -> Result<usize> {
            let total = self.buffer.len();
            // In real implementation, would batch insert
            self.buffer.clear();
            Ok(total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new(DatabaseType::PostgreSQL, "mydb")
            .host("localhost")
            .port(5432)
            .credentials("user", "pass");
        
        assert_eq!(config.database, "mydb");
        assert_eq!(config.host, Some("localhost".to_string()));
        assert_eq!(config.port, Some(5432));
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::select("users")
            .columns(vec!["id", "name", "email"])
            .where_clause("age > 18")
            .order_by("name", false)
            .limit(10);
        
        let sql = query.build_sql();
        assert!(sql.contains("SELECT id, name, email FROM users"));
        assert!(sql.contains("WHERE age > 18"));
        assert!(sql.contains("ORDER BY name ASC"));
        assert!(sql.contains("LIMIT 10"));
    }

    #[test]
    fn test_result_set() {
        let mut result = ResultSet::new(vec!["id".to_string(), "value".to_string()]);
        result.add_row(vec![serde_json::json!(1), serde_json::json!(10.5)]);
        result.add_row(vec![serde_json::json!(2), serde_json::json!(20.3)]);
        
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.column_count(), 2);
        
        let array = result.to_array().unwrap();
        assert_eq!(array.shape(), &[2, 2]);
    }
}