"""
Configuration Management Module

This module provides centralized configuration management for the algo trading system.
It handles loading, validation, and management of all system settings including
API keys, trading parameters, and integration settings.

Classes:
    ConfigManager: Main configuration management class
    APIKeyManager: Manages API keys and tokens
    SettingsValidator: Validates configuration settings

Example:
    >>> from src.config import ConfigManager
    >>> config = ConfigManager()
    >>> config.load_config('config/config.env')
    >>> api_key = config.get_setting('ALPHAVANTAGE_API_KEY')
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
import json


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class APIKeyManager:
    """
    Manages API keys and sensitive tokens.
    
    This class handles the secure loading and validation of API keys
    and tokens required for external service integrations.
    """
    
    def __init__(self) -> None:
        """Initialize the API key manager."""
        self.logger = logging.getLogger(__name__)
        self._api_keys: Dict[str, Optional[str]] = {}
    
    def load_api_key(self, key_name: str, required: bool = True) -> Optional[str]:
        """
        Load an API key from environment variables.
        
        Args:
            key_name: Name of the environment variable containing the API key
            required: Whether the API key is required for system operation
            
        Returns:
            The API key value or None if not found and not required
            
        Raises:
            ConfigurationError: If required API key is missing
        """
        api_key = os.getenv(key_name)
        
        if api_key:
            self._api_keys[key_name] = api_key
            self.logger.info(f"API key loaded: {key_name}")
            return api_key
        elif required:
            raise ConfigurationError(f"Required API key missing: {key_name}")
        else:
            self.logger.warning(f"Optional API key not found: {key_name}")
            return None
    
    def validate_api_key(self, key_name: str, min_length: int = 10) -> bool:
        """
        Validate an API key format.
        
        Args:
            key_name: Name of the API key to validate
            min_length: Minimum required length for the API key
            
        Returns:
            True if API key is valid, False otherwise
        """
        api_key = self._api_keys.get(key_name)
        
        if not api_key:
            return False
        
        if len(api_key) < min_length:
            self.logger.error(f"API key too short: {key_name}")
            return False
        
        # Check for placeholder values
        placeholder_values = [
            'your_api_key_here',
            'your_token_here',
            'your_key_here',
            'placeholder',
            'example'
        ]
        
        if api_key.lower() in placeholder_values:
            self.logger.error(f"API key appears to be placeholder: {key_name}")
            return False
        
        return True
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get an API key value.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            The API key value or None if not found
        """
        return self._api_keys.get(key_name)


class SettingsValidator:
    """
    Validates configuration settings.
    
    This class provides validation methods for different types of
    configuration settings to ensure system reliability.
    """
    
    def __init__(self) -> None:
        """Initialize the settings validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_numeric_setting(
        self, 
        value: Any, 
        setting_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_zero: bool = True
    ) -> bool:
        """
        Validate a numeric setting.
        
        Args:
            value: The value to validate
            setting_name: Name of the setting for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_zero: Whether zero is allowed
            
        Returns:
            True if valid, False otherwise
        """
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            self.logger.error(f"Invalid numeric value for {setting_name}: {value}")
            return False
        
        if not allow_zero and numeric_value == 0:
            self.logger.error(f"Zero not allowed for {setting_name}")
            return False
        
        if min_value is not None and numeric_value < min_value:
            self.logger.error(f"{setting_name} below minimum: {numeric_value} < {min_value}")
            return False
        
        if max_value is not None and numeric_value > max_value:
            self.logger.error(f"{setting_name} above maximum: {numeric_value} > {max_value}")
            return False
        
        return True
    
    def validate_string_setting(
        self, 
        value: Any, 
        setting_name: str,
        allowed_values: Optional[List[str]] = None,
        min_length: int = 1
    ) -> bool:
        """
        Validate a string setting.
        
        Args:
            value: The value to validate
            setting_name: Name of the setting for error messages
            allowed_values: List of allowed values
            min_length: Minimum required length
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, str):
            self.logger.error(f"Invalid string value for {setting_name}: {value}")
            return False
        
        if len(value) < min_length:
            self.logger.error(f"{setting_name} too short: {len(value)} < {min_length}")
            return False
        
        if allowed_values and value not in allowed_values:
            self.logger.error(f"Invalid value for {setting_name}: {value}. Allowed: {allowed_values}")
            return False
        
        return True
    
    def validate_boolean_setting(self, value: Any, setting_name: str) -> bool:
        """
        Validate a boolean setting.
        
        Args:
            value: The value to validate
            setting_name: Name of the setting for error messages
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(value, bool):
            return True
        
        if isinstance(value, str):
            if value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                return True
        
        self.logger.error(f"Invalid boolean value for {setting_name}: {value}")
        return False


class ConfigManager:
    """
    Main configuration management class.
    
    This class provides centralized configuration management including
    loading, validation, and access to all system settings.
    
    Attributes:
        config_path: Path to the configuration file
        settings: Dictionary containing all configuration settings
        api_manager: API key manager instance
        validator: Settings validator instance
    
    Example:
        >>> config = ConfigManager()
        >>> config.load_config('config/config.env')
        >>> if config.validate_config():
        ...     api_key = config.get_setting('ALPHAVANTAGE_API_KEY')
    """
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or 'config/config.env'
        self.settings: Dict[str, Any] = {}
        self.api_manager = APIKeyManager()
        self.validator = SettingsValidator()
        self._validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Define validation rules for configuration settings.
        
        Returns:
            Dictionary containing validation rules for each setting
        """
        return {
            # Trading Strategy Settings
            'RSI_PERIOD': {
                'type': 'numeric',
                'min_value': 5,
                'max_value': 50,
                'required': False,
                'default': 14
            },
            'MA_SHORT_PERIOD': {
                'type': 'numeric',
                'min_value': 5,
                'max_value': 50,
                'required': False,
                'default': 10
            },
            'MA_LONG_PERIOD': {
                'type': 'numeric',
                'min_value': 10,
                'max_value': 200,
                'required': False,
                'default': 20
            },
            'RSI_OVERSOLD': {
                'type': 'numeric',
                'min_value': 10,
                'max_value': 40,
                'required': False,
                'default': 30
            },
            'RSI_OVERBOUGHT': {
                'type': 'numeric',
                'min_value': 60,
                'max_value': 90,
                'required': False,
                'default': 70
            },
            
            # Stock Selection
            'NIFTY_STOCKS': {
                'type': 'string',
                'min_length': 5,
                'required': False,
                'default': 'RELIANCE.NS,HDFCBANK.NS,INFY.NS'
            },
            
            # API Keys
            'ALPHAVANTAGE_API_KEY': {
                'type': 'api_key',
                'required': True,
                'min_length': 10
            },
            'TELEGRAM_BOT_TOKEN': {
                'type': 'api_key',
                'required': False,
                'min_length': 20
            },
            'TELEGRAM_CHAT_ID': {
                'type': 'string',
                'required': False,
                'min_length': 1
            },
            
            # Google Sheets
            'GOOGLE_SHEETS_CREDENTIALS_PATH': {
                'type': 'file_path',
                'required': False,
                'default': 'config/google_credentials.json'
            },
            'GOOGLE_SHEET_ID': {
                'type': 'string',
                'required': False,
                'min_length': 10
            },
            
            # Telegram Alert Settings
            'TELEGRAM_ENABLE_TRADE_ALERTS': {
                'type': 'boolean',
                'required': False,
                'default': True
            },
            'TELEGRAM_ENABLE_ERROR_ALERTS': {
                'type': 'boolean',
                'required': False,
                'default': True
            },
            'TELEGRAM_ENABLE_STATUS_ALERTS': {
                'type': 'boolean',
                'required': False,
                'default': True
            },
            'TELEGRAM_ENABLE_ML_ALERTS': {
                'type': 'boolean',
                'required': False,
                'default': True
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration loaded successfully, False otherwise
            
        Raises:
            ConfigurationError: If configuration file cannot be loaded
        """
        if config_path:
            self.config_path = config_path
        
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            # Load environment variables from file
            load_dotenv(self.config_path)
            
            # Load all settings based on validation rules
            for setting_name, rules in self._validation_rules.items():
                value = os.getenv(setting_name)
                
                if value is None:
                    if rules.get('required', False):
                        raise ConfigurationError(f"Required setting missing: {setting_name}")
                    elif 'default' in rules:
                        value = rules['default']
                        self.logger.info(f"Using default value for {setting_name}: {value}")
                
                if value is not None:
                    # Convert value based on type
                    if rules['type'] == 'numeric':
                        try:
                            value = float(value)
                        except ValueError:
                            self.logger.error(f"Invalid numeric value for {setting_name}: {value}")
                            continue
                    elif rules['type'] == 'boolean':
                        if isinstance(value, str):
                            value = value.lower() in ['true', '1', 'yes', 'on']
                    
                    self.settings[setting_name] = value
            
            # Load API keys
            self._load_api_keys()
            
            self.logger.info(f"Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_api_keys(self) -> None:
        """Load all API keys using the API key manager."""
        api_key_settings = [
            ('ALPHAVANTAGE_API_KEY', True),
            ('TELEGRAM_BOT_TOKEN', False),
        ]
        
        for key_name, required in api_key_settings:
            try:
                self.api_manager.load_api_key(key_name, required)
            except ConfigurationError as e:
                if required:
                    raise
                else:
                    self.logger.warning(str(e))
    
    def validate_config(self) -> bool:
        """
        Validate all configuration settings.
        
        Returns:
            True if all settings are valid, False otherwise
        """
        validation_passed = True
        
        for setting_name, rules in self._validation_rules.items():
            value = self.settings.get(setting_name)
            
            # Skip validation for missing optional settings
            if value is None and not rules.get('required', False):
                continue
            
            # Validate based on type
            if rules['type'] == 'numeric':
                if not self.validator.validate_numeric_setting(
                    value, 
                    setting_name,
                    rules.get('min_value'),
                    rules.get('max_value')
                ):
                    validation_passed = False
            
            elif rules['type'] == 'string':
                if not self.validator.validate_string_setting(
                    value,
                    setting_name,
                    rules.get('allowed_values'),
                    rules.get('min_length', 1)
                ):
                    validation_passed = False
            
            elif rules['type'] == 'boolean':
                if not self.validator.validate_boolean_setting(value, setting_name):
                    validation_passed = False
            
            elif rules['type'] == 'api_key':
                if not self.api_manager.validate_api_key(
                    setting_name,
                    rules.get('min_length', 10)
                ):
                    validation_passed = False
            
            elif rules['type'] == 'file_path':
                if value and not Path(value).exists():
                    self.logger.warning(f"File not found for {setting_name}: {value}")
        
        if validation_passed:
            self.logger.info("Configuration validation passed")
        else:
            self.logger.error("Configuration validation failed")
        
        return validation_passed
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting value.
        
        Args:
            key: Setting name
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a configuration setting.
        
        Args:
            key: Setting name
            value: New value
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate the new value if validation rules exist
            if key in self._validation_rules:
                rules = self._validation_rules[key]
                
                if rules['type'] == 'numeric':
                    if not self.validator.validate_numeric_setting(
                        value, key, rules.get('min_value'), rules.get('max_value')
                    ):
                        return False
                elif rules['type'] == 'string':
                    if not self.validator.validate_string_setting(
                        value, key, rules.get('allowed_values'), rules.get('min_length', 1)
                    ):
                        return False
                elif rules['type'] == 'boolean':
                    if not self.validator.validate_boolean_setting(value, key):
                        return False
            
            self.settings[key] = value
            self.logger.info(f"Setting updated: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating setting {key}: {e}")
            return False
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Get trading-specific configuration settings.
        
        Returns:
            Dictionary containing trading configuration
        """
        return {
            'rsi_period': self.get_setting('RSI_PERIOD', 14),
            'ma_short_period': self.get_setting('MA_SHORT_PERIOD', 10),
            'ma_long_period': self.get_setting('MA_LONG_PERIOD', 20),
            'rsi_oversold': self.get_setting('RSI_OVERSOLD', 30),
            'rsi_overbought': self.get_setting('RSI_OVERBOUGHT', 70),
            'stocks': self.get_setting('NIFTY_STOCKS', 'RELIANCE.NS,HDFCBANK.NS,INFY.NS').split(',')
        }
    
    def get_api_config(self) -> Dict[str, Optional[str]]:
        """
        Get API configuration settings.
        
        Returns:
            Dictionary containing API configuration
        """
        return {
            'alphavantage_api_key': self.api_manager.get_api_key('ALPHAVANTAGE_API_KEY'),
            'telegram_bot_token': self.api_manager.get_api_key('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': self.get_setting('TELEGRAM_CHAT_ID'),
        }
    
    def get_integration_config(self) -> Dict[str, Any]:
        """
        Get integration configuration settings.
        
        Returns:
            Dictionary containing integration configuration
        """
        return {
            'google_sheets_credentials_path': self.get_setting('GOOGLE_SHEETS_CREDENTIALS_PATH'),
            'google_sheet_id': self.get_setting('GOOGLE_SHEET_ID'),
            'telegram_alerts': {
                'trade_alerts': self.get_setting('TELEGRAM_ENABLE_TRADE_ALERTS', True),
                'error_alerts': self.get_setting('TELEGRAM_ENABLE_ERROR_ALERTS', True),
                'status_alerts': self.get_setting('TELEGRAM_ENABLE_STATUS_ALERTS', True),
                'ml_alerts': self.get_setting('TELEGRAM_ENABLE_ML_ALERTS', True),
            }
        }
    
    def export_config(self, output_path: str) -> bool:
        """
        Export current configuration to a file.
        
        Args:
            output_path: Path to output file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            config_data = {
                'settings': self.settings,
                'validation_rules': self._validation_rules
            }
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Configuration exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            'config_file': self.config_path,
            'total_settings': len(self.settings),
            'api_keys_loaded': len([k for k in self.settings.keys() if 'API_KEY' in k or 'TOKEN' in k]),
            'trading_config': self.get_trading_config(),
            'integrations_enabled': {
                'google_sheets': bool(self.get_setting('GOOGLE_SHEET_ID')),
                'telegram': bool(self.get_setting('TELEGRAM_BOT_TOKEN')),
            }
        }