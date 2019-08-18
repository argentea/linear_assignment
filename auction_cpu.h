/**
 * @file   auction_cpu.h
 * @author Jiaqi Gu, Yibo Lin
 * @date   Apr 2019
 */
#include <iostream>
#include <cstring>
#include <cassert>

#define AUCTION_MAX_EPS 10.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.1
#define BIG_NEGATIVE -9999999

template <typename T>
int run_auction(
    int    num_nodes,
    T* data_ptr,      // data, num_nodes*num_nodes in row-major  
    T*   person2item_ptr, // results
    
    float auction_max_eps,
    float auction_min_eps,
    float auction_factor
)
{
    
    // --
    // Declare variables
    
    int  *num_assigned_ptr   = (int*)malloc(sizeof(int));
    int* item2person_ptr = (int*)malloc(num_nodes * sizeof(int));
    T* bids_ptr      = (T*)malloc(num_nodes * num_nodes * sizeof(T));
    T* prices_ptr    = (T*)malloc(num_nodes * sizeof(T));
    //int* bidders_ptr     = (int*)malloc(num_nodes * num_nodes * sizeof(int)); // unused
    int* sbids_ptr       = (int*)malloc(num_nodes * sizeof(int));

    int   *data           = data_ptr;
    int   &num_assigned   = *(num_assigned_ptr);
    int   *person2item    = person2item_ptr;
    int   *item2person    = item2person_ptr;
    int   *prices         = prices_ptr;
    int   *sbids          = sbids_ptr;
    int   *bids           = bids_ptr;

    for(int i = 0; i < num_nodes; i++) {
        prices[i] = 0.0;
        person2item[i] = -1;
    }

    // Start timer


    float auction_eps = auction_max_eps;
    int counter = 0;
    while(auction_eps >= auction_min_eps) {
        num_assigned = 0;

        for(int i = 0; i < num_nodes; i++) {
            person2item[i] = -1;
            item2person[i] = -1;
        }
        num_assigned = 0;


        while(num_assigned < num_nodes){
            counter += 1;

            std::memset(bids, 0, num_nodes * num_nodes * sizeof(T));
            std::memset(sbids, 0, num_nodes * sizeof(int));

            // #pragma omp parallel for num_threads(1)
            for(int i = 0; i < num_nodes; i++) {
                if(person2item[i] == -1) {
                    T top1_val = BIG_NEGATIVE; 
                    T top2_val = BIG_NEGATIVE; 
                    int top1_col; 
                    T tmp_val;

                    for (int col = 0; col < num_nodes; col++)
                    {
                        tmp_val = data[i * num_nodes + col]; 
                        if (tmp_val < 0)
                        {
                            continue;
                        }
                        tmp_val = tmp_val - prices[col];
                        if (tmp_val >= top1_val)
                        {
                            top2_val = top1_val;
                            top1_col = col;
                            top1_val = tmp_val;
                        }
                        else if (tmp_val > top2_val)
                        {
                            top2_val = tmp_val;
                        }
                    }
                    if (top2_val == BIG_NEGATIVE)
                    {
                        top2_val = top1_val;
                    }
                    T bid = top1_val - top2_val + auction_eps;
                    bids[num_nodes * top1_col + i] = bid;
                    sbids[top1_col] = 1; 
#if 0
                            int top1_col   = 0;
                            int top1_val = data[i*num_nodes] - prices[top1_col];

                            int top2_val = -1000;
                            
                            int col;
                            int tmp_val;
                            for (col = 0; col < num_nodes; ++col){
                                tmp_val = data[i*num_nodes+col] - prices[col];
                                
                                if(tmp_val > top1_val){
                                    top2_val = top1_val;
                                    
                                    top1_col = col;
                                    top1_val = tmp_val;
                                } else if(tmp_val > top2_val){
                                    top2_val = tmp_val;
                                }        
                            }
                            
                            int bid = top1_val - top2_val + auction_eps;
                            bids[num_nodes * top1_col + i] = bid;
                            sbids[top1_col] = 1;
#endif
                }
            }

            // #pragma omp parallel for num_threads(1)
            for(int j = 0; j < num_nodes; j++) {
                if(sbids[j] != 0) {
                    T high_bid  = 0.0;
                    int high_bidder = -1;

                    T tmp_bid = -1;
                    for(int i = 0; i < num_nodes; i++){
                        tmp_bid = bids[num_nodes * j + i]; 
                        if(tmp_bid > high_bid){
                            high_bid    = tmp_bid;
                            high_bidder = i;
                        }
                    }
                    int current_person = item2person[j];
                    if(current_person >= 0){
                        person2item[current_person] = -1; 
                    } else {
                        // #pragma omp atomic
                        num_assigned++;
                    }

                    prices[j]                += high_bid;
                    person2item[high_bidder] = j;
                    item2person[j]           = high_bidder;
                }
            }
        }

        auction_eps *= auction_factor;
    } 

    // //Print results
    // int score = 0;
    // for (int i = 0; i < num_nodes; i++) {
    //     std::cout << i << " " << person2item[i] << std::endl;
    //     score += data[i * num_nodes + person2item[i]];
    // }
    // std::cerr << "score=" <<score << std::endl;   

    free(num_assigned_ptr); 
    free(item2person_ptr); 
    free(bids_ptr);
    free(prices_ptr);  
    //free(bidders_ptr); 
    free(sbids_ptr); 

    return 0;
} // end run_auction

/// @brief solve assignment problem with auction algorithm 
/// The matrix is converted to non-negative matrix with maximization 
/// Skipped edges are assigned with BIG_NEGATIVE
/// @param cost a nxn row-major cost matrix 
/// @param sol solution mapping from row to column 
/// @param n dimension 
/// @param skip_threshold if the weight is larger than the threshold, do not add the edge 
/// @param minimize_flag true for minimization problem and false or maximization 
template <typename T>
T AuctionAlgorithmCPULauncher(const T* cost, int* sol, int n, T skip_threshold = std::numeric_limits<T>::max(), 
        int minimize_flag = 1, 
        T auction_max_eps=AUCTION_MAX_EPS,
        T auction_min_eps=AUCTION_MIN_EPS, 
        float auction_factor=AUCTION_FACTOR
        )
{
    int nn = n*n; 
    T* matrix = new T [n*n]; 
    if (minimize_flag)
    {
        T max_cost = 0; 
        for (int i = 0; i < nn; ++i)
        {
            T c = cost[i];
            if (c < skip_threshold)
            {
                max_cost = std::max(max_cost, c); 
            }
        }
        for ( int row = 0 ; row < n ; row++ ) 
        {
            for ( int col = 0 ; col < n ; col++ ) 
            {
                int idx = n*row+col;
                T c = cost[idx]; 
                matrix[idx] = (c < skip_threshold)? max_cost-c : BIG_NEGATIVE;
            }
        }
    }
    else 
    {
        std::copy(cost, cost+nn, matrix); 
    }

    run_auction(
        n,
        matrix,
        sol,
        auction_max_eps,
        auction_min_eps,
        auction_factor
    );

	// Get solution and display objective.
    T total_cost = 0; 
	for ( int row = 0 ; row < n ; row++ ) 
    {
        int col = sol[row]; 
        total_cost += cost[n*row+col]; 
	}

    delete [] matrix; 

    return total_cost; 
}
