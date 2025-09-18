#ifndef CONFIG_H
#define CONFIG_H

#include "face_mask_detector.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration file sections
#define CONFIG_SECTION_GENERAL "general"
#define CONFIG_SECTION_DETECTION "detection"
#define CONFIG_SECTION_MODELS "models"
#define CONFIG_SECTION_UI "ui"
#define CONFIG_SECTION_LOGGING "logging"

// Default configuration values
#define DEFAULT_CONFIG_FILE "config/face_mask_detector.conf"
#define DEFAULT_MODEL_DIR "models"
#define DEFAULT_CASCADE_FILE "models/haarcascade_frontalface_alt.xml"
#define DEFAULT_MASK_MODEL_FILE "models/mask_detector.onnx"
#define DEFAULT_LOG_FILE "logs/face_mask_detector.log"

// Logging levels
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR = 3,
    LOG_LEVEL_FATAL = 4,
    LOG_LEVEL_NONE = 5
} log_level_t;

// Configuration structure for logging
typedef struct {
    log_level_t level;
    char log_file[MAX_PATH_LENGTH];
    bool console_output;
    bool file_output;
    bool timestamp_enabled;
    bool thread_id_enabled;
    size_t max_file_size;
    int max_backup_files;
} logging_config_t;

// Extended application configuration
typedef struct {
    app_config_t app;
    logging_config_t logging;
    char config_file_path[MAX_PATH_LENGTH];
    time_t last_modified;
    bool auto_reload;
} extended_config_t;

// Configuration validation structure
typedef struct {
    bool valid;
    char error_message[MAX_STRING_LENGTH];
    int error_count;
    char warnings[10][MAX_STRING_LENGTH];
    int warning_count;
} config_validation_t;

// Configuration functions
int load_configuration_file(const char* config_path, extended_config_t* config);
int save_configuration_file(const char* config_path, const extended_config_t* config);
int reload_configuration_if_changed(extended_config_t* config);
void set_default_extended_config(extended_config_t* config);

// Configuration validation
config_validation_t validate_configuration(const extended_config_t* config);
bool validate_file_path(const char* path, bool must_exist);
bool validate_directory_path(const char* path, bool must_exist);
bool validate_numeric_range(double value, double min_val, double max_val);

// Configuration file parsing
int parse_config_line(const char* line, char* section, char* key, char* value);
int parse_boolean_value(const char* value);
double parse_double_value(const char* value, double default_val);
int parse_integer_value(const char* value, int default_val);

// Environment variable support
const char* get_config_from_env(const char* env_var, const char* default_value);
void apply_environment_overrides(extended_config_t* config);

// Configuration templates
int create_default_config_file(const char* config_path);
int create_config_template(const char* template_path);

// Configuration utilities
void print_configuration_summary(const extended_config_t* config);
int backup_configuration(const char* config_path);
int restore_configuration_backup(const char* config_path);

// Logging configuration functions
int init_logging_system(const logging_config_t* config);
void cleanup_logging_system(void);
int set_log_level(log_level_t level);
log_level_t string_to_log_level(const char* level_str);
const char* log_level_to_string(log_level_t level);

// Configuration monitoring
typedef struct {
    char config_path[MAX_PATH_LENGTH];
    time_t last_check;
    bool monitoring_enabled;
    void (*change_callback)(const extended_config_t* old_config, const extended_config_t* new_config);
} config_monitor_t;

int init_config_monitor(config_monitor_t* monitor, const char* config_path,
                       void (*callback)(const extended_config_t*, const extended_config_t*));
void cleanup_config_monitor(config_monitor_t* monitor);
int check_config_changes(config_monitor_t* monitor, extended_config_t* config);

// Configuration export/import
int export_config_to_json(const extended_config_t* config, const char* output_path);
int import_config_from_json(const char* input_path, extended_config_t* config);

// INI file utilities
typedef struct {
    char** sections;
    char** keys;
    char** values;
    int count;
    int capacity;
} ini_data_t;

int load_ini_file(const char* path, ini_data_t* ini);
void cleanup_ini_data(ini_data_t* ini);
const char* get_ini_value(const ini_data_t* ini, const char* section, const char* key, const char* default_val);
int set_ini_value(ini_data_t* ini, const char* section, const char* key, const char* value);
int save_ini_file(const char* path, const ini_data_t* ini);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H