// Neural Network Class
//   author: Andrew Gohlich
//   data: 4/25/2026

#ifndef NN_H
#define NN_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

class NN {
private:
    std::vector<int> layers; // stores neural net layers defined at instantiation

    std::vector<std::vector<double>> x_matrix; // n x input_dimension
    std::vector<std::vector<double>> y_matrix; // n x output_dimension

    std::vector<std::vector<std::vector<double>>> weights; // [L][j][k] = w^(L)_jk
    std::vector<std::vector<double>> biases; // [L][j] = b^(L)_j

    std::vector<std::vector<double>> activations; // [L][j] = a^(L)_j

public:
    // CTOR - specifies neural net layers and data
    NN(const std::vector<int> &layers, 
       const std::vector<std::vector<double>> &x, 
       const std::vector<std::vector<double>> &y) 
     : layers(layers), x_matrix(x), y_matrix(y), weights(layers.size()), biases(layers.size()), activations(layers.size()) {
        
        std::random_device rd;  // Non-deterministic seed source
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    
        for(size_t i = 1; i<layers.size(); ++i){
            std::normal_distribution<double> distrib(0, sqrt(1.0 / layers[i-1]));

            weights[i] = std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i-1])); // randomly initialize weight
            for(size_t j = 0; j<layers[i]; ++j)
                for(size_t k = 0; k<layers[i-1]; ++k)
                    weights[i][j][k] = distrib(gen);
            biases[i] = std::vector<double>(layers[i], 0); // initialize bias to 0

            activations[i] = std::vector<double>(layers[i], 0);
        }
        weights[0] = {};
        biases[0] = std::vector<double>(layers[0], 0);
        activations[0] = std::vector<double>(layers[0], 0);
    }


    // HELPER FUNCTIONS
    // ----------------
    // Outputs info about NN object
    void about(){
        std::cout << "Layers: ";
        for(int i:layers) std::cout << i << " ";
        std::cout << '\n';
        std::cout << "X Matrix - " << x_matrix.size() << " x " << x_matrix[0].size() << '\n';
        std::cout << "Y Matrix - " << y_matrix.size() << " x " << y_matrix[0].size() << '\n';
    }

    // Computes sigmoid function 
    double sigmoid(double d){
        return 1.0 / (1 + exp(-d));
    }

    // Computes MSE loss
    double MSE(const std::vector<double> &y_pred, const std::vector<double> &y_actual){
        double tot = 0;

        for(size_t i = 0; i<y_pred.size(); ++i){
            tot += (y_pred[i]-y_actual[i])*(y_pred[i]-y_actual[i]);
        }

        return tot / y_pred.size();
    }


    // CORE FUNCTIONS
    // --------------
    // Runs forward Neural Network on given inputs
    std::vector<double> predict(const std::vector<double> &x_input){
        size_t N = x_input.size();
        
        // check correct size
        if(x_input.size() != x_matrix[0].size()){
            std::cerr << "Error: cannot predict because input vectors are wrong dimension - "<<x_input.size()<<"\n";
            exit(1);
        }

        // looping through each sample
        std::vector<double> out(layers.back());

        // input layer
        for(size_t j = 0; j<x_input.size(); ++j){
            activations[0][j] = x_input[j];
        }

        // subsequent layers
        for(size_t l = 1; l<layers.size(); ++l){
            for(size_t j = 0; j<layers[l]; ++j){
                activations[l][j] = biases[l][j];

                for(size_t k = 0; k<layers[l-1]; ++k){
                    activations[l][j] += activations[l-1][k] * weights[l][j][k];
                }

                activations[l][j] = sigmoid(activations[l][j]);
            }
        }

        // extracting output layer
        for(size_t j = 0; j<layers.back(); ++j){
            out[j] = activations.back()[j];
        }

        return out;
    }

    // Trains model, returns loss log
    std::vector<double> train(size_t epochs, double lr, size_t log_rate = 10){
        // setup
        std::vector<double> log;
        log.reserve(epochs/log_rate);
        std::random_device rd;  // Non-deterministic seed source
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distrib(0, x_matrix.size()-1); // random int distribution between 0 and 149 inclusive

        // loop through epochs
        for(size_t epoch = 0; epoch<epochs; ++epoch){

            // INITIALIZING GRADIENTS TO 0
            // ---------------------------
            double epoch_loss = 0.0;
            std::vector<std::vector<std::vector<double>>> weights_update(weights.size());
            std::vector<std::vector<double>> biases_update(biases.size());

            for(size_t l = 1; l<layers.size(); ++l){
                weights_update[l] = std::vector<std::vector<double>>(layers[l], std::vector<double>(layers[l-1], 0.0));
                biases_update[l] = std::vector<double>(layers[l], 0.0);
            }


            // LOOPING OVER ALL TRAINING SAMPLES 
            // ---------------------------------
            for(size_t i = 0; i<x_matrix.size(); ++i){

                // selecting datapoint
                std::vector<double> x_train = x_matrix[i];
                std::vector<double> y_train = y_matrix[i];

                // predicting - running neural network forward
                std::vector<double> y_pred = predict(x_train);
                epoch_loss += MSE(y_pred, y_train);

                // backpropagation
                //  -> initializing current training sample gradients to 0 
                std::vector<std::vector<std::vector<double>>> dW(weights.size());
                std::vector<std::vector<double>> dB(biases.size());

                for(size_t l = 1; l<layers.size(); ++l){
                    dW[l] = std::vector<std::vector<double>>(layers[l], std::vector<double>(layers[l-1], 0.0));
                    dB[l] = std::vector<double>(layers[l], 0.0);
                }

                //  -> output layer
                size_t l = layers.size()-1;
                for(size_t j = 0; j<layers[l]; ++j){
                    for(size_t k = 0; k<layers[l-1]; ++k){
                        dW[l][j][k] = 2 * (activations[l][j] - y_train[j]) * activations[l][j]*(1-activations[l][j]) * activations[l-1][k];
                    }
                    dB[l][j] = 2 * (activations[l][j] - y_train[j]) * activations[l][j]*(1-activations[l][j]);
                }

                //  -> hidden layers
                for(l = l-1; l>0; --l){
                    for(size_t k = 0; k<layers[l]; ++k){
                        double sum = 0;
                        for(size_t j = 0; j<layers[l+1]; ++j){
                            sum += dB[l+1][j] * weights[l+1][j][k];
                        }
                        dB[l][k] = sum * activations[l][k]*(1-activations[l][k]);

                        for(size_t j = 0; j<layers[l-1]; ++j){
                            dW[l][k][j] = dB[l][k] * activations[l-1][j];
                        }
                    }
                }

                // -> accumulate gradient
                for(size_t l = 1; l<layers.size(); ++l){
                    for(size_t j = 0; j<layers[l]; ++j){
                        for(size_t k = 0; k<layers[l-1]; ++k){
                            weights_update[l][j][k] += dW[l][j][k];
                        }
                        biases_update[l][j] += dB[l][j];
                    }
                }
            }


            // AVERAGE GRADIENTS
            // -----------------
            size_t N = x_matrix.size();
            for(size_t l = 1; l<layers.size(); ++l){
                for(size_t j = 0; j<layers[l]; ++j){
                    for(size_t k = 0; k<layers[l-1]; ++k){
                        weights_update[l][j][k] /= N;
                    }
                    biases_update[l][j] /= N;
                }
            }


            // UPDATE
            // ------
            for(size_t l = 1; l<layers.size(); ++l){
                for(size_t j = 0; j<layers[l]; ++j){
                    for(size_t k = 0; k<layers[l-1]; ++k){
                        weights[l][j][k] -= lr * weights_update[l][j][k];
                    }
                    biases[l][j] -= lr * biases_update[l][j];
                }
            }

            // LOSS LOG
            // --------
            if(epoch % log_rate == 0){
                log.push_back(epoch_loss / N);
                std::cout << "\tepoch - " << epoch << ", mse - " << log.back() << '\n';
            }
        }

        return log;
    }

};

#endif