/**
 * @file   auction_cpu_.cpp
 * @author Jiaqi Gu, Yibo Lin
 * @date   Apr 2019
 */
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include "auction_cpu.h"

#define NUM_NODES 128 
#define BATCH_SIZE 1024 

int load_data(int *raw_data) {
    std::ifstream input_file("./graph4", std::ios_base::in);
    
    //std::cerr << "load_data: start" << std::endl;
    int i = 0;
    int val;
    while(input_file >> val) {
        raw_data[i] = val;
        i++;
    }
    //std::cerr << "load_data: finish" << std::endl;
    return (int)sqrt(i);
}

int main(int argc, char **argv)
{
    int NUM_THREADS = 10; 
    if (argc > 1)
    {
        NUM_THREADS = atoi(argv[1]);
    }
    std::cerr << "use " << NUM_THREADS << " threads for " << BATCH_SIZE << " graphs\n";
    std::cerr << "num_nodes=" << NUM_NODES << std::endl;
    
    // Load data
    int num_graphs = BATCH_SIZE;
    int num_nodes = NUM_NODES;
    int* raw_data = (int *)malloc(sizeof(int) * num_nodes * num_nodes * num_graphs);
    
    for (int i = 0; i < num_graphs; ++i)
    {
        int num_nodes = load_data(raw_data + i*num_nodes*num_nodes);
    }
    int num_edges = num_nodes * num_nodes;
    if(num_nodes <= 0) {
        return 1;
    }
    
    int* h_data  = (int *)realloc(raw_data, sizeof(int) * num_nodes * num_nodes * num_graphs);

    // convert to minimization problem 
    int max_weight; 
    for (int i = 0; i < num_graphs; ++i)
    {
        max_weight = 0; 
        for (int row = 0; row < num_nodes; ++row)
        {
            for (int col = 0; col < num_nodes; ++col)
            {
                max_weight = std::max(max_weight, h_data[i*num_nodes*num_nodes + row*num_nodes + col]); 
            }
        }
        for (int j = 0; j < num_nodes*num_nodes; ++j)
        {
            h_data[i*num_nodes*num_nodes + j] = max_weight - h_data[i*num_nodes*num_nodes + j];
        }
    }
    
    // Dense
    int* h_offsets = (int *)malloc(num_graphs* sizeof(int) * (num_nodes + 1));
    h_offsets[0] = 0;
    for(int j = 0;j < num_graphs;j++){
        for(int i = 1; i < num_nodes + 1; i++) {
            h_offsets[j*(num_nodes+1) + i] = i * num_nodes;
        }
    }

    int* h_columns = (int *)malloc(num_graphs*sizeof(int) * num_nodes*num_nodes);
    for(int j = 0;j < num_graphs;j++){
        for(int i = 0; i < num_nodes*num_nodes; i++) {
            h_columns[j*num_nodes*num_nodes+ i] = i % num_nodes;
        }
    }
    
    int* h_person2item = (int *)malloc(sizeof(int) * num_nodes * num_graphs);
    
    int verbose = 1;
    int cost; 
    
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads (NUM_THREADS) 
    for (int i = 0; i < num_graphs; ++i)
    {
        cost = AuctionAlgorithmCPULauncher(
                h_data + i*num_nodes*num_nodes,
                h_person2item + i*num_nodes,
                num_nodes, 
                std::numeric_limits<int>::max(), 
                1
                );
    }
    // Stop timer
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    std::cerr << "milliseconds=" << 1000 * time_span.count() << std::endl;            
    std::cerr << "minimization problem cost = " << cost << std::endl; 
    std::cerr << "maximization problem score = " << max_weight*NUM_NODES-cost << std::endl; 

    // Print results
    // int score = 0;
    // for (int i = 0; i < num_nodes; i++) {
    //     std::cout << i << " " << h_person2item[i] << std::endl;
    //     // score += h_data[i + num_nodes * h_person2item[i]];
    //     score += h_data[i * num_nodes + h_person2item[i]];
    // }
    
    // std::cerr << "score=" << (int)score << std::endl;        

    free(h_data);
    free(h_offsets);
    free(h_columns);
    free(h_person2item);
}

