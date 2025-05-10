#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define IMAGE_SIZE 28  
#define TOTAL_IMAGES_A 100  
#define TOTAL_IMAGES_B 100  
#define TOTAL_IMAGES (TOTAL_IMAGES_A + TOTAL_IMAGES_B)
#define TRAINING_PERCENTAGE 80  
#define TEST_PERCENTAGE (100 - TRAINING_PERCENTAGE) 
#define VECTOR_SIZE (IMAGE_SIZE * IMAGE_SIZE + 1)
#define MAX_TRAIN_SAMPLES 160
#define MAX_TEST_SAMPLES 40
#define LEARNING_RATE_GD 0.0001f
#define LEARNING_RATE_SGD 0.01f
#define LEARNING_RATE_ADAM 0.01f   
#define BETA1 0.9      
#define BETA2 0.999    
#define EPSILON 1e-8
#define WCOUNT 5

typedef struct {
    float pixels[IMAGE_SIZE][IMAGE_SIZE];
} Image;

typedef struct {
    float vector[VECTOR_SIZE];
    int label; 
} DataPoint;

double get_current_time_in_seconds();
int read_image(const char* filename, Image* image);
void generate_filename(char* filename, const char* class_name, int index);
void image_to_vector(Image* img, float* vector);
void split_data(int* indices, int* training_indices, int* test_indices, int* training_count, int* test_count);
int parse_data_split(const char* split_file_path, DataPoint* train_data, int* train_size, 
                     DataPoint* test_data, int* test_size, 
                     float vectors[][IMAGE_SIZE * IMAGE_SIZE + 1]);
float calculate_loss(const DataPoint* data, int size, const float* weights);
float compute_accuracy(const DataPoint* test_data, int test_size, const float* weights);
void calculate_gradients(const DataPoint* data, int size, const float* weights, float* gradients);
void calculate_gradient_single(const DataPoint* dp, const float* weights, float* gradient);
void gradient_descent(DataPoint* train_data, int train_size, 
                      DataPoint* test_data, int test_size, 
                      float* weights, int epochs, const char* output_file, const char* weight_file);
void stochastic_gradient_descent(DataPoint* train_data, int train_size, 
                                 DataPoint* test_data, int test_size, 
                                 float* weights, int epochs, const char* output_file, const char* weight_file);
void adam_optimizer(DataPoint* train_data, int train_size, 
                    DataPoint* test_data, int test_size, 
                    float* weights, int epochs, const char* output_file, const char* weight_file);
void save_weights(float* weigths, int size, const char* output_file);                    

int main() {

    // Bellek alanları
    Image images[TOTAL_IMAGES];
    DataPoint train_data[MAX_TRAIN_SAMPLES];
    DataPoint test_data[MAX_TEST_SAMPLES];
    float vectors[TOTAL_IMAGES][IMAGE_SIZE * IMAGE_SIZE + 1];
    float initial_weights[5][VECTOR_SIZE];
    int indices[TOTAL_IMAGES];
    int training_indices[TOTAL_IMAGES];
    int test_indices[TOTAL_IMAGES];
    int training_count = 0, test_count = 0, train_size = 0, test_size = 0;

    // Dosya yolları
    const char* outputs_gd[5] = {
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w1/gradient_descent1.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w2/gradient_descent2.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w3/gradient_descent3.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w4/gradient_descent4.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w5/gradient_descent5.txt"
    };
    const char* outputs_sgd[5] = {
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w1/stochastic_gradient_descent1.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w2/stochastic_gradient_descent2.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w3/stochastic_gradient_descent3.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w4/stochastic_gradient_descent4.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w5/stochastic_gradient_descent5.txt",
    };
    const char* outputs_adam[5] = {
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w1/adam1.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w2/adam2.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w3/adam3.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w4/adam4.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data/w5/adam5.txt"
    };
    const char* updated_weights[3] = {
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/updated_weights/gd_updated.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/updated_weights/sgd_updated.txt",
        "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/updated_weights/adam_updated.txt"
    };

    // A ve B görüntülerini okuma
    char filename[200];
    for (int i = 0; i < TOTAL_IMAGES_A; i++) {
        generate_filename(filename, "class_A", i + 1);
        if (!read_image(filename, &images[i])) {
            printf("Error reading image: %s\n", filename);
            return 1;
        }
    }
    for (int i = 0; i < TOTAL_IMAGES_B; i++) {
        generate_filename(filename, "class_B", i + 1);
        if (!read_image(filename, &images[TOTAL_IMAGES_A + i])) {
            printf("Error reading image: %s\n", filename);
            return 1;
        }
    }

    // Eğitim ve test kümelerine ayırma işlemi
    split_data(indices, training_indices, test_indices, &training_count, &test_count);

    // Tüm görüntüleri vektöre dönüştürme işlemi
    for (int i = 0; i < TOTAL_IMAGES; i++) {
        image_to_vector(&images[i], vectors[i]);
    }

    // Eğitim ve test kümesindeki verilerin etiket ve vektörünü içeren elemanlara dönüştürülmesi 
    if (!parse_data_split("data_split.txt", train_data, &train_size, test_data, &test_size, vectors)) {
        return 1;
    }

    // İlk w değerleri ataması
    for (int i=0; i<VECTOR_SIZE; i++) {
        initial_weights[0][i] = -2.0;
        initial_weights[1][i] = 0.0;
        initial_weights[2][i] = 0.01;
        initial_weights[3][i] = -0.02;
        initial_weights[4][i] = -0.002;
    }

    // Optimizasyon
    for (int i=0; i< 5; i++) {
        printf("Initial weights = %.5f\n", initial_weights[i][0]);
        gradient_descent(train_data, train_size, test_data, test_size, initial_weights[i], 100, outputs_gd[i], updated_weights[0]);
        stochastic_gradient_descent(train_data, train_size, test_data, test_size, initial_weights[i], 100, outputs_sgd[i], updated_weights[1]);
        adam_optimizer(train_data, train_size, test_data, test_size, initial_weights[i], 100, outputs_adam[i], updated_weights[2]);
    }
    return 0;
}

double get_current_time_in_seconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int read_image(const char* filename, Image* image) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file: %s\n", filename);
        return 0;
    }

    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            fscanf(file, "%f", &image->pixels[i][j]);
        }
    }

    fclose(file);
    return 1;
}

void generate_filename(char* filename, const char* class_name, int index) {
    sprintf(filename, "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/images_detailed/%s_image_%03d.txt", class_name, index);
}

void image_to_vector(Image* img, float* vector) {
    int idx = 0;
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            vector[idx++] = img->pixels[i][j];
        }
    }
    vector[idx] = 1.0f; 
}

void split_data(int* indices, int* training_indices, int* test_indices, int* training_count, int* test_count) {
    FILE* split_file = fopen("data_split.txt", "r");
    if (!split_file) {
        printf("File not found, generating new split data...\n");
        
        for (int i = 0; i < TOTAL_IMAGES; i++) {
            indices[i] = i + 1;
        }
        for (int i = TOTAL_IMAGES - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        *training_count = (TOTAL_IMAGES * TRAINING_PERCENTAGE) / 100;
        *test_count = TOTAL_IMAGES - *training_count;

        for (int i = 0; i < *training_count; i++) {
            training_indices[i] = indices[i];
        }
        for (int i = 0; i < *test_count; i++) {
            test_indices[i] = indices[*training_count + i];
        }

        split_file = fopen("data_split.txt", "w");
        if (!split_file) {
            printf("Error creating data_split.txt\n");
            return;
        }

        fprintf(split_file, "Train:\n");
        for (int i = 0; i < *training_count; i++) {
            fprintf(split_file, "%s%d\n", training_indices[i] <= TOTAL_IMAGES_A ? "A" : "B",
                    training_indices[i] <= TOTAL_IMAGES_A ? training_indices[i] : training_indices[i] - TOTAL_IMAGES_A);
        }

        fprintf(split_file, "Test:\n");
        for (int i = 0; i < *test_count; i++) {
            fprintf(split_file, "%s%d\n", test_indices[i] <= TOTAL_IMAGES_A ? "A" : "B",
                    test_indices[i] <= TOTAL_IMAGES_A ? test_indices[i] : test_indices[i] - TOTAL_IMAGES_A);
        }

        fclose(split_file);
        printf("Data split completed. Training set: %d images, Test set: %d images.\n", training_count, test_count);
    } 
    else {
    printf("Reading existing split data...\n");
    char line[256];
    int i = 0, j = 0;
    int is_in_training = 0;
    int is_in_test = 0;
    
    while (fgets(line, sizeof(line), split_file)) {
        if (strncmp(line, "Train:", 6) == 0) {
            is_in_training = 1;
            is_in_test = 0;  
            continue;
        }
        if (strncmp(line, "Test:", 5) == 0) {
            is_in_test = 1;
            is_in_training = 0;  
            continue;
        }
        
        if (is_in_training) {
            if (line[0] == 'A') {
                training_indices[i++] = atoi(&line[1]); 
            } else if (line[0] == 'B') {
                training_indices[i++] = TOTAL_IMAGES_A + atoi(&line[1]); 
            }
        }
        
        if (is_in_test) {
            if (line[0] == 'A') {
                test_indices[j++] = atoi(&line[1]); 
            } else if (line[0] == 'B') {
                test_indices[j++] = TOTAL_IMAGES_A + atoi(&line[1]); 
            }
        }
    }
    fclose(split_file);
}

}

int parse_data_split(const char* split_file_path, DataPoint* train_data, int* train_size, 
                     DataPoint* test_data, int* test_size, 
                     float vectors[][IMAGE_SIZE * IMAGE_SIZE + 1]) {
    FILE* file = fopen(split_file_path, "r");
    if (!file) {
        printf("Error opening %s\n", split_file_path);
        return 0;
    }

    char line[256];
    int index;
    char class_label;
    int vector_index;

    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "Train:", 6) == 0) {
            while (fgets(line, sizeof(line), file) && strncmp(line, "Test:", 5) != 0) {

                sscanf(line, "%c%d", &class_label, &index);

                DataPoint* dp = &train_data[(*train_size)++];

                dp->label = (class_label == 'A') ? 1 : -1;

                if (class_label == 'A') {
                    vector_index = index - 1;  
                } else if (class_label == 'B') {
                    vector_index = TOTAL_IMAGES_A + (index - 1); 
                }

                for (int i = 0; i < VECTOR_SIZE; i++) {
                    dp->vector[i] = vectors[vector_index][i];
                }
            }
        } else  {
            while (fgets(line, sizeof(line), file)) {

                sscanf(line, "%c%d", &class_label, &index);

                DataPoint* dp = &test_data[(*test_size)++];

                dp->label = (class_label == 'A') ? 1 : -1;

                if (class_label == 'A') {
                    vector_index = index - 1;  
                } else if (class_label == 'B') {
                    vector_index = TOTAL_IMAGES_A + (index - 1);  
                }

                for (int i = 0; i < VECTOR_SIZE; i++) {
                    dp->vector[i] = vectors[vector_index][i];
                }
            }
        }
    }

    fclose(file);
    return 1;
}

float calculate_loss(const DataPoint* data, int size, const float* weights) {
    float total_loss = 0.0f;

    for (int i = 0; i < size; i++) {
        const DataPoint* dp = &data[i];
        float wx = 0.0f;

        for (int j = 0; j < VECTOR_SIZE; j++) {
            wx += weights[j] * dp->vector[j];
        }

        float y_pred = tanh(wx);
        float error = dp->label - y_pred;
        total_loss += error * error;
    }
    return total_loss / size;
}

void calculate_gradients(const DataPoint* data, int size, const float* weights, float* gradients) {
    memset(gradients, 0, sizeof(float) * VECTOR_SIZE);

    for (int i = 0; i < size; i++) {
        const DataPoint* dp = &data[i];
        float wx = 0.0f;

        for (int j = 0; j < VECTOR_SIZE; j++) {
            wx += weights[j] * dp->vector[j];
        }

        float y_pred = tanh(wx);
        float error = dp->label - y_pred;
        float scale = -2 * error * (1 - y_pred * y_pred); 

        for (int j = 0; j < VECTOR_SIZE; j++) {
            gradients[j] += scale * dp->vector[j];
        }
    }
}

void calculate_gradient_single(const DataPoint* dp, const float* weights, float* gradient) {
    float wx = 0.0f;

    for (int j = 0; j < VECTOR_SIZE; j++) {
        wx += weights[j] * dp->vector[j];
    }

    float y_pred = tanh(wx);
    float error = dp->label - y_pred;
    float scale = -2 * error * (1 - y_pred * y_pred);

    for (int j = 0; j < VECTOR_SIZE; j++) {
        gradient[j] = scale * dp->vector[j];
    }
}

float compute_accuracy(const DataPoint* test_data, int test_size, const float* weights) {
    int correct_predictions = 0;

    for (int i = 0; i < test_size; i++) {
        float wx = 0.0f;
        for (int j = 0; j < VECTOR_SIZE; j++) {
            wx += test_data[i].vector[j] * weights[j];
        }

        float prediction = tanh(wx);

        int predicted_label = (prediction >= 0) ? 1 : -1;

        if (predicted_label == test_data[i].label) {
            correct_predictions++;
        }
    }
    return (float)correct_predictions / test_size;
}

void gradient_descent(DataPoint* train_data, int train_size, 
                      DataPoint* test_data, int test_size, 
                      float* weights, int epochs, const char* output_file, const char* weight_file) {
    float gradients[VECTOR_SIZE];
    FILE* file = fopen(output_file, "w");
    if (!file) {
        printf("Error opening %s\n", output_file);
    }
    double total_time = 0.0f;

    fprintf(file, "Epoch, Loss, Train Accuracy, Test Accuracy, Time Taken(s)\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        double start_time = get_current_time_in_seconds();

        float loss = calculate_loss(train_data, train_size, weights);

        calculate_gradients(train_data, train_size, weights, gradients);

        for (int j = 0; j < VECTOR_SIZE; j++) {
            weights[j] -= LEARNING_RATE_GD * gradients[j];
        }

        float test_accuracy = compute_accuracy(test_data, test_size, weights);
        float train_accuracy = compute_accuracy(train_data, train_size, weights);

        double end_time = get_current_time_in_seconds();

        double epoch_time = end_time - start_time;
        total_time += epoch_time; 

        fprintf(file, "%d,%.6f,%.2f,%.2f,%.6f\n", 
                epoch + 1, loss, train_accuracy, test_accuracy, total_time);
        save_weights(weights, VECTOR_SIZE, weight_file);
    }
    fclose(file);
    printf("Gradient Descent optimization completed. Results saved to %s\n", output_file);
}

void stochastic_gradient_descent(DataPoint* train_data, int train_size, 
                                 DataPoint* test_data, int test_size, 
                                 float* weights, int epochs, const char* output_file, const char* weight_file) {
    FILE* file = fopen(output_file, "w");
    if (!file) {
        printf("Error opening %s\n", output_file);
        return;
    }

    srand(time(NULL)); 

    double total_time = 0.0f;
    fprintf(file, "Epoch, Loss, Train Accuracy, Test Accuracy, Time Taken(s)\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        double start_epoch_time = get_current_time_in_seconds();

        for (int i = 0; i < train_size; i++) {
            int random_index = rand() % train_size;
            const DataPoint* dp = &train_data[random_index];
            float gradient[VECTOR_SIZE];

            calculate_gradient_single(dp, weights, gradient);

            for (int j = 0; j < VECTOR_SIZE; j++) {
                weights[j] -= LEARNING_RATE_SGD * gradient[j];
            }
        }

        float loss = calculate_loss(train_data, train_size, weights);

        float test_accuracy = compute_accuracy(test_data, test_size, weights);
        float train_accuracy = compute_accuracy(train_data, train_size, weights);

        double epoch_time = get_current_time_in_seconds() - start_epoch_time;
        total_time += epoch_time;

        fprintf(file, "%d,%.6f,%.2f,%.2f,%.6f\n", 
                epoch + 1, loss, train_accuracy, test_accuracy, total_time);

        save_weights(weights, VECTOR_SIZE, weight_file);
    }
    fclose(file);
    printf("Stochastic Gradient Descent optimization completed. Results saved to %s\n", output_file);
}


void adam_optimizer(DataPoint* train_data, int train_size, 
                    DataPoint* test_data, int test_size, 
                    float* weights, int epochs, const char* output_file, const char* weight_file) {
    float m[VECTOR_SIZE] = {0}; 
    float v[VECTOR_SIZE] = {0}; 
    float gradients[VECTOR_SIZE];
    float m_hat[VECTOR_SIZE], v_hat[VECTOR_SIZE]; 
    int t = 0; 
    double total_time = 0.0f;

    FILE* file = fopen(output_file, "w");
    if (!file) {
        printf("Error opening %s\n", output_file);
    }

    fprintf(file, "Epoch, Loss, Train Accuracy, Test Accuracy, Time Taken(s)\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        double start_time = get_current_time_in_seconds();

        t++; 

        calculate_gradients(train_data, train_size, weights, gradients);

        for (int i = 0; i < VECTOR_SIZE; i++) {
            m[i] = BETA1 * m[i] + (1 - BETA1) * gradients[i];
            v[i] = BETA2 * v[i] + (1 - BETA2) * gradients[i] * gradients[i];
            m_hat[i] = m[i] / (1 - pow(BETA1, t));
            v_hat[i] = v[i] / (1 - pow(BETA2, t));
            weights[i] -= LEARNING_RATE_ADAM * m_hat[i] / (sqrt(v_hat[i]) + EPSILON);
        }

        float loss = calculate_loss(train_data, train_size, weights);
        float train_accuracy = compute_accuracy(train_data, train_size, weights);
        float test_accuracy = compute_accuracy(test_data, test_size, weights);

        double end_time = get_current_time_in_seconds();
        double epoch_time = end_time - start_time;
        total_time += epoch_time;

        fprintf(file, "%d,%.6f,%.2f,%.2f,%.6f\n", 
                epoch + 1, loss, train_accuracy, test_accuracy, total_time);
        save_weights(weights, VECTOR_SIZE, weight_file);
    }
    fclose(file);
    printf("Adam optimization completed. Results saved to %s\n", output_file);
}

void save_weights(float* weights, int size, const char* output_file) {
    FILE* check_file = fopen(output_file, "r");
    if (check_file) {
        fclose(check_file);
        return;
    }

    FILE* file = fopen(output_file, "a");
    if (!file) {
        return;
    }

    for (int i = 0; i < size; i++) {
        fprintf(file, "%.6f,", weights[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}
