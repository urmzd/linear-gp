//! Tracing configuration and initialization for the LGP framework.
//!
//! This module provides utilities for setting up structured logging and tracing
//! throughout the Linear GP system.
//!
//! # Usage
//!
//! ```rust,no_run
//! use lgp::utils::tracing::{TracingConfig, init_tracing};
//!
//! // Initialize with defaults (reads RUST_LOG and LGP_LOG_FORMAT env vars)
//! init_tracing(TracingConfig::default());
//!
//! // Or with custom configuration
//! use lgp::utils::tracing::TracingFormat;
//! let config = TracingConfig::new()
//!     .with_format(TracingFormat::Json)
//!     .with_span_events(true);
//! init_tracing(config);
//! ```
//!
//! # Environment Variables
//!
//! - `RUST_LOG`: Controls log level filtering (e.g., `lgp=debug`, `lgp=trace`)
//! - `LGP_LOG_FORMAT`: Override output format (`pretty`, `compact`, `json`)

use std::env;
use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    prelude::*,
    EnvFilter,
};

/// Output format for tracing logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TracingFormat {
    /// Human-readable, colorized output with full span information.
    /// Best for development and debugging.
    #[default]
    Pretty,
    /// Condensed single-line output.
    /// Good for production with moderate verbosity.
    Compact,
    /// JSON-structured output.
    /// Best for log aggregation systems (ELK, Datadog, etc.).
    Json,
}

impl std::str::FromStr for TracingFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pretty" => Ok(TracingFormat::Pretty),
            "compact" => Ok(TracingFormat::Compact),
            "json" => Ok(TracingFormat::Json),
            _ => Err(format!(
                "Unknown format: {}. Expected: pretty, compact, or json",
                s
            )),
        }
    }
}

impl TracingFormat {
    /// Parse format from string (case-insensitive), returning None if invalid.
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// Configuration for tracing initialization.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Output format for logs.
    pub format: TracingFormat,
    /// Whether to log span enter/exit events.
    pub span_events: bool,
    /// Whether to include file name and line numbers in output.
    pub file_info: bool,
    /// Whether to include thread IDs in output.
    pub thread_ids: bool,
    /// Whether to include thread names in output.
    pub thread_names: bool,
    /// Whether to include target (module path) in output.
    pub target: bool,
    /// Default filter directive if RUST_LOG is not set.
    pub default_filter: String,
    /// Optional log file path. If set, logs are written to this file.
    pub log_file: Option<PathBuf>,
    /// Whether to also log to stdout when file logging is enabled.
    pub log_to_stdout: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            format: TracingFormat::Pretty,
            span_events: false,
            file_info: false,
            thread_ids: false,
            thread_names: false,
            target: true,
            default_filter: "lgp=info".to_string(),
            log_file: None,
            log_to_stdout: true,
        }
    }
}

impl TracingConfig {
    /// Create a new tracing configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the output format.
    pub fn with_format(mut self, format: TracingFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable or disable span enter/exit events.
    pub fn with_span_events(mut self, enabled: bool) -> Self {
        self.span_events = enabled;
        self
    }

    /// Enable or disable file name and line number output.
    pub fn with_file_info(mut self, enabled: bool) -> Self {
        self.file_info = enabled;
        self
    }

    /// Enable or disable thread ID output.
    pub fn with_thread_ids(mut self, enabled: bool) -> Self {
        self.thread_ids = enabled;
        self
    }

    /// Enable or disable thread name output.
    pub fn with_thread_names(mut self, enabled: bool) -> Self {
        self.thread_names = enabled;
        self
    }

    /// Enable or disable target (module path) output.
    pub fn with_target(mut self, enabled: bool) -> Self {
        self.target = enabled;
        self
    }

    /// Set the default filter directive (used if RUST_LOG is not set).
    pub fn with_default_filter(mut self, filter: impl Into<String>) -> Self {
        self.default_filter = filter.into();
        self
    }

    /// Set log file path (enables file logging).
    pub fn with_log_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_file = Some(path.into());
        self
    }

    /// Control whether to also log to stdout when file logging is enabled.
    pub fn with_stdout(mut self, enabled: bool) -> Self {
        self.log_to_stdout = enabled;
        self
    }

    /// Create a configuration optimized for verbose debugging.
    pub fn verbose() -> Self {
        Self {
            format: TracingFormat::Pretty,
            span_events: true,
            file_info: true,
            thread_ids: true,
            thread_names: false,
            target: true,
            default_filter: "lgp=debug".to_string(),
            log_file: None,
            log_to_stdout: true,
        }
    }

    /// Create a configuration optimized for production/JSON logging.
    pub fn production() -> Self {
        Self {
            format: TracingFormat::Json,
            span_events: false,
            file_info: false,
            thread_ids: false,
            thread_names: false,
            target: true,
            default_filter: "lgp=info".to_string(),
            log_file: None,
            log_to_stdout: true,
        }
    }
}

/// Initialize the tracing subscriber with the given configuration.
///
/// Returns a `WorkerGuard` if file logging is enabled. This guard must be held
/// for the duration of the program to ensure all logs are flushed to the file.
///
/// This function should be called once at application startup, before any
/// tracing macros are used.
///
/// # Environment Variables
///
/// - `RUST_LOG`: Controls log level filtering. Examples:
///   - `lgp=debug` - Debug level for lgp crate
///   - `lgp=trace` - Trace level for lgp crate (very verbose)
///   - `lgp::core=trace,lgp=info` - Different levels for different modules
///
/// - `LGP_LOG_FORMAT`: Override the output format regardless of config.
///   Values: `pretty`, `compact`, `json`
///
/// # Panics
///
/// This function will panic if called more than once, as the global subscriber
/// can only be set once.
pub fn init_tracing(config: TracingConfig) -> Option<WorkerGuard> {
    // Check for format override via environment variable
    let format = env::var("LGP_LOG_FORMAT")
        .ok()
        .and_then(|s| TracingFormat::parse(&s))
        .unwrap_or(config.format);

    // Build the environment filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.default_filter));

    // Determine span events
    let span_events = if config.span_events {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    // If file logging is configured, use non-blocking file writer
    if let Some(log_path) = &config.log_file {
        // Create parent directories if needed
        if let Some(parent) = log_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }

        // Create file appender with non-blocking writer
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .expect("Failed to open log file");

        let (non_blocking, guard) = tracing_appender::non_blocking(file);

        // Build subscriber with file layer (and optionally stdout)
        if config.log_to_stdout {
            // Both file and stdout
            init_with_file_and_stdout(format, filter, span_events, &config, non_blocking);
        } else {
            // File only
            init_with_file_only(format, filter, span_events, &config, non_blocking);
        }

        return Some(guard);
    }

    // Standard stdout-only setup
    init_stdout_only(format, filter, span_events, &config);
    None
}

/// Initialize tracing with file output only.
fn init_with_file_only(
    format: TracingFormat,
    filter: EnvFilter,
    span_events: FmtSpan,
    config: &TracingConfig,
    writer: tracing_appender::non_blocking::NonBlocking,
) {
    match format {
        TracingFormat::Pretty => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .with_writer(writer)
                    .with_ansi(false)
                    .pretty()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Compact => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .with_writer(writer)
                    .with_ansi(false)
                    .compact()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Json => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .with_writer(writer)
                    .json()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
    }
}

/// Initialize tracing with both file and stdout output.
fn init_with_file_and_stdout(
    format: TracingFormat,
    filter: EnvFilter,
    span_events: FmtSpan,
    config: &TracingConfig,
    file_writer: tracing_appender::non_blocking::NonBlocking,
) {
    match format {
        TracingFormat::Pretty => {
            let file_layer = fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .pretty()
                .with_span_events(span_events.clone())
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let stdout_layer = fmt::layer()
                .pretty()
                .with_span_events(span_events)
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(file_layer)
                .with(stdout_layer);
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Compact => {
            let file_layer = fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .compact()
                .with_span_events(span_events.clone())
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let stdout_layer = fmt::layer()
                .compact()
                .with_span_events(span_events)
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(file_layer)
                .with(stdout_layer);
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Json => {
            let file_layer = fmt::layer()
                .with_writer(file_writer)
                .json()
                .with_span_events(span_events.clone())
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let stdout_layer = fmt::layer()
                .json()
                .with_span_events(span_events)
                .with_file(config.file_info)
                .with_line_number(config.file_info)
                .with_thread_ids(config.thread_ids)
                .with_thread_names(config.thread_names)
                .with_target(config.target);
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(file_layer)
                .with(stdout_layer);
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
    }
}

/// Initialize tracing with stdout only.
fn init_stdout_only(
    format: TracingFormat,
    filter: EnvFilter,
    span_events: FmtSpan,
    config: &TracingConfig,
) {
    match format {
        TracingFormat::Pretty => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .pretty()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Compact => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .compact()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
        TracingFormat::Json => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .json()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");
        }
    }
}

/// Try to initialize tracing, returning Ok if successful or if already initialized.
///
/// This is useful in tests or when multiple initialization paths exist.
pub fn try_init_tracing(config: TracingConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Check for format override via environment variable
    let format = env::var("LGP_LOG_FORMAT")
        .ok()
        .and_then(|s| TracingFormat::parse(&s))
        .unwrap_or(config.format);

    // Build the environment filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.default_filter));

    // Determine span events
    let span_events = if config.span_events {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    // Build and set the subscriber based on format
    let result = match format {
        TracingFormat::Pretty => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .pretty()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
        }
        TracingFormat::Compact => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .compact()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
        }
        TracingFormat::Json => {
            let subscriber = tracing_subscriber::registry().with(filter).with(
                fmt::layer()
                    .json()
                    .with_span_events(span_events)
                    .with_file(config.file_info)
                    .with_line_number(config.file_info)
                    .with_thread_ids(config.thread_ids)
                    .with_thread_names(config.thread_names)
                    .with_target(config.target),
            );
            tracing::subscriber::set_global_default(subscriber)
        }
    };

    result.map_err(|e| e.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_from_str() {
        assert_eq!(TracingFormat::parse("pretty"), Some(TracingFormat::Pretty));
        assert_eq!(TracingFormat::parse("PRETTY"), Some(TracingFormat::Pretty));
        assert_eq!(
            TracingFormat::parse("compact"),
            Some(TracingFormat::Compact)
        );
        assert_eq!(TracingFormat::parse("json"), Some(TracingFormat::Json));
        assert_eq!(TracingFormat::parse("invalid"), None);
    }

    #[test]
    fn test_config_builder() {
        let config = TracingConfig::new()
            .with_format(TracingFormat::Json)
            .with_span_events(true)
            .with_file_info(true)
            .with_thread_ids(true)
            .with_default_filter("lgp=trace");

        assert_eq!(config.format, TracingFormat::Json);
        assert!(config.span_events);
        assert!(config.file_info);
        assert!(config.thread_ids);
        assert_eq!(config.default_filter, "lgp=trace");
    }

    #[test]
    fn test_verbose_config() {
        let config = TracingConfig::verbose();
        assert_eq!(config.format, TracingFormat::Pretty);
        assert!(config.span_events);
        assert!(config.file_info);
        assert_eq!(config.default_filter, "lgp=debug");
    }

    #[test]
    fn test_production_config() {
        let config = TracingConfig::production();
        assert_eq!(config.format, TracingFormat::Json);
        assert!(!config.span_events);
        assert!(!config.file_info);
        assert_eq!(config.default_filter, "lgp=info");
    }

    #[test]
    fn test_file_logging_config() {
        let config = TracingConfig::new()
            .with_log_file("/tmp/test.log")
            .with_stdout(false);

        assert_eq!(config.log_file, Some(PathBuf::from("/tmp/test.log")));
        assert!(!config.log_to_stdout);

        // Default should have no log file and stdout enabled
        let default = TracingConfig::default();
        assert!(default.log_file.is_none());
        assert!(default.log_to_stdout);
    }
}
