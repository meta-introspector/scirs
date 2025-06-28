//! Cloud storage integration for datasets
//!
//! This module provides functionality for loading datasets from various cloud storage providers:
//! - Amazon S3
//! - Google Cloud Storage (GCS)
//! - Azure Blob Storage
//! - Generic S3-compatible storage

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cache::DatasetCache;
use crate::error::{DatasetsError, Result};
use crate::external::ExternalClient;
use crate::utils::Dataset;

/// Configuration for cloud storage access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Cloud provider type
    pub provider: CloudProvider,
    /// Region (for AWS/GCS)
    pub region: Option<String>,
    /// Bucket name
    pub bucket: String,
    /// Access credentials
    pub credentials: CloudCredentials,
    /// Custom endpoint URL (for S3-compatible services)
    pub endpoint: Option<String>,
    /// Whether to use virtual-hosted-style URLs
    pub path_style: bool,
    /// Custom headers
    pub headers: HashMap<String, String>,
}

/// Supported cloud storage providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudProvider {
    /// Amazon S3
    S3,
    /// Google Cloud Storage
    GCS,
    /// Azure Blob Storage
    Azure,
    /// Generic S3-compatible storage
    S3Compatible,
}

/// Cloud storage credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloudCredentials {
    /// Access key and secret
    AccessKey {
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
    },
    /// Service account key (GCS)
    ServiceAccount { key_file: String },
    /// Azure storage account key
    AzureKey {
        account_name: String,
        account_key: String,
    },
    /// Use environment variables
    Environment,
    /// Anonymous access
    Anonymous,
}

/// Cloud storage client
pub struct CloudClient {
    config: CloudConfig,
    cache: DatasetCache,
    external_client: ExternalClient,
}

impl CloudClient {
    /// Create a new cloud client
    pub fn new(config: CloudConfig) -> Result<Self> {
        let cache = DatasetCache::new()?;
        let external_client = ExternalClient::new()?;

        Ok(Self {
            config,
            cache,
            external_client,
        })
    }

    /// Load a dataset from cloud storage
    pub fn load_dataset(&self, key: &str) -> Result<Dataset> {
        // Check cache first
        let cache_key = format!("cloud_{}_{}", self.config.bucket, key);
        if let Ok(cached_data) = self.cache.get(&cache_key) {
            return self.parse_cached_data(&cached_data);
        }

        // Build the URL based on provider
        let url = self.build_url(key)?;

        // Load using external client with authentication
        let mut external_config = crate::external::ExternalConfig::default();
        self.add_authentication_headers(&mut external_config)?;

        let external_client = ExternalClient::with_config(external_config)?;
        let dataset = external_client.download_dataset_sync(&url, None)?;

        // Cache the result
        if let Ok(serialized) = serde_json::to_vec(&dataset) {
            let _ = self.cache.put(&cache_key, &serialized);
        }

        Ok(dataset)
    }

    /// List objects in a cloud storage bucket with a prefix
    pub fn list_datasets(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        match self.config.provider {
            CloudProvider::S3 | CloudProvider::S3Compatible => self.list_s3_objects(prefix),
            CloudProvider::GCS => self.list_gcs_objects(prefix),
            CloudProvider::Azure => self.list_azure_objects(prefix),
        }
    }

    /// Upload a dataset to cloud storage
    #[allow(dead_code)]
    pub fn upload_dataset(&self, key: &str, dataset: &Dataset) -> Result<()> {
        let serialized =
            serde_json::to_vec(dataset).map_err(|e| DatasetsError::SerdeError(e.to_string()))?;

        self.upload_data(key, &serialized, "application/json")
    }

    /// Build URL for the given key
    fn build_url(&self, key: &str) -> Result<String> {
        match self.config.provider {
            CloudProvider::S3 => {
                let region = self.config.region.as_deref().unwrap_or("us-east-1");
                if self.config.path_style {
                    Ok(format!(
                        "https://s3.{}.amazonaws.com/{}/{}",
                        region, self.config.bucket, key
                    ))
                } else {
                    Ok(format!(
                        "https://{}.s3.{}.amazonaws.com/{}",
                        self.config.bucket, region, key
                    ))
                }
            }
            CloudProvider::S3Compatible => {
                let endpoint = self.config.endpoint.as_ref().ok_or_else(|| {
                    DatasetsError::InvalidFormat(
                        "S3-compatible storage requires endpoint".to_string(),
                    )
                })?;

                if self.config.path_style {
                    Ok(format!("{}/{}/{}", endpoint, self.config.bucket, key))
                } else {
                    Ok(format!(
                        "https://{}.{}/{}",
                        self.config.bucket,
                        endpoint.trim_start_matches("https://"),
                        key
                    ))
                }
            }
            CloudProvider::GCS => Ok(format!(
                "https://storage.googleapis.com/{}/{}",
                self.config.bucket, key
            )),
            CloudProvider::Azure => {
                let account_name = match &self.config.credentials {
                    CloudCredentials::AzureKey { account_name, .. } => account_name,
                    _ => {
                        return Err(DatasetsError::InvalidFormat(
                            "Azure requires account name in credentials".to_string(),
                        ))
                    }
                };
                Ok(format!(
                    "https://{}.blob.core.windows.net/{}/{}",
                    account_name, self.config.bucket, key
                ))
            }
        }
    }

    /// Add authentication headers based on credentials
    fn add_authentication_headers(
        &self,
        config: &mut crate::external::ExternalConfig,
    ) -> Result<()> {
        match (&self.config.provider, &self.config.credentials) {
            (
                CloudProvider::S3 | CloudProvider::S3Compatible,
                CloudCredentials::AccessKey {
                    access_key,
                    secret_key,
                    session_token,
                },
            ) => {
                // For simplicity, we'll use presigned URLs or implement basic auth
                // In a real implementation, you'd use proper AWS signature v4
                config.headers.insert(
                    "Authorization".to_string(),
                    format!("AWS {}:{}", access_key, secret_key),
                );

                if let Some(token) = session_token {
                    config
                        .headers
                        .insert("X-Amz-Security-Token".to_string(), token.clone());
                }
            }
            (CloudProvider::GCS, CloudCredentials::ServiceAccount { key_file }) => {
                // For GCS, you'd typically use OAuth 2.0 with JWT
                // This is a simplified approach
                config.headers.insert(
                    "Authorization".to_string(),
                    format!("Bearer {}", self.get_gcs_token(key_file)?),
                );
            }
            (
                CloudProvider::Azure,
                CloudCredentials::AzureKey {
                    account_name,
                    account_key,
                },
            ) => {
                // Azure uses shared key authentication
                let auth_header = self.create_azure_auth_header(account_name, account_key)?;
                config
                    .headers
                    .insert("Authorization".to_string(), auth_header);
            }
            (_, CloudCredentials::Anonymous) => {
                // No authentication needed
            }
            (_, CloudCredentials::Environment) => {
                // Use environment variables - in a real implementation, you'd read from env
                return Err(DatasetsError::AuthenticationError(
                    "Environment credentials not implemented".to_string(),
                ));
            }
            _ => {
                return Err(DatasetsError::AuthenticationError(
                    "Invalid credential type for provider".to_string(),
                ));
            }
        }

        // Add custom headers
        for (key, value) in &self.config.headers {
            config.headers.insert(key.clone(), value.clone());
        }

        Ok(())
    }

    fn parse_cached_data(&self, data: &[u8]) -> Result<Dataset> {
        serde_json::from_slice(data).map_err(|e| DatasetsError::SerdeError(e.to_string()))
    }

    #[allow(dead_code)]
    fn get_gcs_token(&self, _key_file: &str) -> Result<String> {
        // Simplified token generation - in reality, you'd implement full OAuth 2.0 flow
        Err(DatasetsError::AuthenticationError(
            "GCS token generation not implemented".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn create_azure_auth_header(&self, _account_name: &str, _account_key: &str) -> Result<String> {
        // Simplified Azure auth - in reality, you'd implement proper shared key signature
        Err(DatasetsError::AuthenticationError(
            "Azure authentication not implemented".to_string(),
        ))
    }

    fn list_s3_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let list_url = match self.config.provider {
            CloudProvider::S3 => {
                let region = self.config.region.as_deref().unwrap_or("us-east-1");
                format!(
                    "https://s3.{}.amazonaws.com/{}/?list-type=2",
                    region, self.config.bucket
                )
            }
            CloudProvider::S3Compatible => {
                let endpoint = self.config.endpoint.as_ref().ok_or_else(|| {
                    DatasetsError::InvalidFormat(
                        "S3-compatible storage requires endpoint".to_string(),
                    )
                })?;
                format!("{}/{}/?list-type=2", endpoint, self.config.bucket)
            }
            _ => unreachable!(),
        };

        let url_with_prefix = if let Some(prefix) = prefix {
            format!("{}&prefix={}", list_url, prefix)
        } else {
            list_url
        };

        // For simplicity, we'll return an error indicating this needs proper implementation
        // In reality, you'd parse the XML response from S3
        Err(DatasetsError::Other(
            "S3 listing not implemented - requires XML parsing".to_string(),
        ))
    }

    fn list_gcs_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let list_url = format!(
            "https://storage.googleapis.com/storage/v1/b/{}/o",
            self.config.bucket
        );

        let url_with_prefix = if let Some(prefix) = prefix {
            format!("{}?prefix={}", list_url, prefix)
        } else {
            list_url
        };

        // For simplicity, we'll return an error indicating this needs proper implementation
        // In reality, you'd parse the JSON response from GCS API
        Err(DatasetsError::Other(format!(
            "GCS listing not implemented - would call {}",
            url_with_prefix
        )))
    }

    fn list_azure_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let account_name = match &self.config.credentials {
            CloudCredentials::AzureKey { account_name, .. } => account_name,
            _ => {
                return Err(DatasetsError::InvalidFormat(
                    "Azure requires account name".to_string(),
                ))
            }
        };

        let list_url = format!(
            "https://{}.blob.core.windows.net/{}?restype=container&comp=list",
            account_name, self.config.bucket
        );

        let url_with_prefix = if let Some(prefix) = prefix {
            format!("{}&prefix={}", list_url, prefix)
        } else {
            list_url
        };

        // For simplicity, we'll return an error indicating this needs proper implementation
        // In reality, you'd parse the XML response from Azure
        Err(DatasetsError::Other(format!(
            "Azure listing not implemented - would call {}",
            url_with_prefix
        )))
    }

    #[allow(dead_code)]
    fn upload_data(&self, key: &str, data: &[u8], content_type: &str) -> Result<()> {
        let url = self.build_url(key)?;

        // For uploads, you'd need to implement PUT requests with proper authentication
        // This is a simplified placeholder
        Err(DatasetsError::Other(format!(
            "Upload not implemented for key: {} to URL: {}, content-type: {}",
            key, url, content_type
        )))
    }
}

/// Pre-configured cloud clients for popular services
pub mod presets {
    use super::*;

    /// Create an S3 client with access key credentials
    pub fn s3_client(
        region: &str,
        bucket: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some(region.to_string()),
            bucket: bucket.to_string(),
            credentials: CloudCredentials::AccessKey {
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                session_token: None,
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create a GCS client with service account credentials
    pub fn gcs_client(bucket: &str, key_file: &str) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::GCS,
            region: None,
            bucket: bucket.to_string(),
            credentials: CloudCredentials::ServiceAccount {
                key_file: key_file.to_string(),
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an Azure Blob Storage client
    pub fn azure_client(
        account_name: &str,
        account_key: &str,
        container: &str,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::Azure,
            region: None,
            bucket: container.to_string(),
            credentials: CloudCredentials::AzureKey {
                account_name: account_name.to_string(),
                account_key: account_key.to_string(),
            },
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an S3-compatible client (e.g., MinIO, DigitalOcean Spaces)
    pub fn s3_compatible_client(
        endpoint: &str,
        bucket: &str,
        access_key: &str,
        secret_key: &str,
        path_style: bool,
    ) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3Compatible,
            region: None,
            bucket: bucket.to_string(),
            credentials: CloudCredentials::AccessKey {
                access_key: access_key.to_string(),
                secret_key: secret_key.to_string(),
                session_token: None,
            },
            endpoint: Some(endpoint.to_string()),
            path_style,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }

    /// Create an anonymous S3 client for public buckets
    pub fn public_s3_client(region: &str, bucket: &str) -> Result<CloudClient> {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some(region.to_string()),
            bucket: bucket.to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        CloudClient::new(config)
    }
}

/// Popular public datasets available in cloud storage
pub mod public_datasets {
    use super::presets::*;
    use super::*;

    /// AWS Open Data sets
    pub struct AWSOpenData;

    impl AWSOpenData {
        /// Load the Common Crawl dataset (sample)
        pub fn common_crawl_sample() -> Result<CloudClient> {
            public_s3_client("us-east-1", "commoncrawl")
        }

        /// Load NOAA weather data
        pub fn noaa_weather() -> Result<CloudClient> {
            public_s3_client("us-east-1", "noaa-global-hourly-pds")
        }

        /// Load NASA Landsat data
        pub fn nasa_landsat() -> Result<CloudClient> {
            public_s3_client("us-west-2", "landsat-pds")
        }

        /// Load NYC Taxi data
        pub fn nyc_taxi() -> Result<CloudClient> {
            public_s3_client("us-east-1", "nyc-tlc")
        }
    }

    /// Google Cloud Public Datasets
    pub struct GCPPublicData;

    impl GCPPublicData {
        /// Load BigQuery public datasets (requires authentication)
        pub fn bigquery_samples(key_file: &str) -> Result<CloudClient> {
            gcs_client("bigquery-public-data", key_file)
        }

        /// Load Google Books Ngrams
        pub fn books_ngrams(key_file: &str) -> Result<CloudClient> {
            gcs_client("books", key_file)
        }
    }

    /// Microsoft Azure Open Datasets
    pub struct AzureOpenData;

    impl AzureOpenData {
        /// Load COVID-19 tracking data
        pub fn covid19_tracking(account_name: &str, account_key: &str) -> Result<CloudClient> {
            azure_client(account_name, account_key, "covid19-tracking")
        }

        /// Load US Census data
        pub fn us_census(account_name: &str, account_key: &str) -> Result<CloudClient> {
            azure_client(account_name, account_key, "us-census")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::presets::*;
    use super::*;

    #[test]
    fn test_cloud_config_creation() {
        let config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some("us-east-1".to_string()),
            bucket: "test-bucket".to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: false,
            headers: HashMap::new(),
        };

        assert!(matches!(config.provider, CloudProvider::S3));
        assert_eq!(config.bucket, "test-bucket");
    }

    #[test]
    fn test_s3_url_building() {
        let client = public_s3_client("us-east-1", "test-bucket").unwrap();
        let url = client.build_url("path/to/dataset.csv").unwrap();
        assert_eq!(
            url,
            "https://test-bucket.s3.us-east-1.amazonaws.com/path/to/dataset.csv"
        );
    }

    #[test]
    fn test_s3_path_style_url() {
        let mut config = CloudConfig {
            provider: CloudProvider::S3,
            region: Some("us-east-1".to_string()),
            bucket: "test-bucket".to_string(),
            credentials: CloudCredentials::Anonymous,
            endpoint: None,
            path_style: true,
            headers: HashMap::new(),
        };

        let client = CloudClient::new(config).unwrap();
        let url = client.build_url("test.csv").unwrap();
        assert_eq!(
            url,
            "https://s3.us-east-1.amazonaws.com/test-bucket/test.csv"
        );
    }

    #[test]
    fn test_gcs_url_building() {
        let client = gcs_client("test-bucket", "dummy-key.json").unwrap();
        let url = client.build_url("data/file.json").unwrap();
        assert_eq!(
            url,
            "https://storage.googleapis.com/test-bucket/data/file.json"
        );
    }

    #[test]
    fn test_azure_url_building() {
        let client = azure_client("testaccount", "dummykey", "container").unwrap();
        let url = client.build_url("blob.txt").unwrap();
        assert_eq!(
            url,
            "https://testaccount.blob.core.windows.net/container/blob.txt"
        );
    }

    #[test]
    fn test_s3_compatible_url_building() {
        let client = s3_compatible_client(
            "https://minio.example.com",
            "my-bucket",
            "access",
            "secret",
            true,
        )
        .unwrap();

        let url = client.build_url("file.csv").unwrap();
        assert_eq!(url, "https://minio.example.com/my-bucket/file.csv");
    }

    #[test]
    fn test_aws_open_data_clients() {
        // Test that we can create public dataset clients
        let result = public_datasets::AWSOpenData::noaa_weather();
        assert!(result.is_ok());

        let result = public_datasets::AWSOpenData::nyc_taxi();
        assert!(result.is_ok());
    }
}
