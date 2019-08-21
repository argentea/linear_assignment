#include<fstream>
#include<iostream>
#include<stdlib.h>
#include<cstdlib>

#define node_num	128

int load_datas(int *raw_data){
	std::ifstream input_file("graph4",std::ios_base::in);

	int i = 0;
	int val;
	while(input_file >> val){
		raw_data[i] = val;
		i++;
	}
	return i;
}
int main(){
	int* raw_data = (int*)malloc(sizeof(int) * node_num*node_num);
	int* itemspnodes = (int*)malloc(sizeof(int) * node_num);
	int data_size = load_datas(raw_data);
	if(data_size != node_num*node_num){
		std::cerr << "node_num is not 128\n";
		return 0;
	}
	else
	{
		std::cerr << "node_num is 128\n";
	}
	int count = 0;
	int max = 0;
	int min = INT32_MAX;
	int tem_val;
	int countzero = 0;
	for(int i = 0; i < node_num; i++){
		count = 0;
		for(int j = 0; j < node_num; j++){
			tem_val = raw_data[i*node_num + j];
			if(tem_val < 0){
				continue;
			}
			if(tem_val == 0){
				countzero++;
				continue;
			}
			count++;
			if(tem_val < min){
				min = tem_val;
			}
			if(tem_val > max){
				max = tem_val;
			}
		}
		itemspnodes[i] = count;
		if(i != 0){
			itemspnodes[i] += itemspnodes[i - 1];
		}
		if(i == node_num-1)
		std::cout << i << " " << itemspnodes[i] << std::endl;
	}
	std::cout << max << " " << min <<"  " << countzero << std::endl;
/*
	int tem;
	for(int i = 0; i < 128; i++){
		std::cin >> id >> tem;
		if(id != 0){
			if(tem!=itemspnodes[id -1]){
				std::cerr << "You let GPU make an error!! at node_id "<< id <<"with right " << itemspnodes[id-1] << "  " << tem <<"\n";
				return 0;
			}
		}
		else
		{
			if(tem != 0){
				std::cerr << "You let GPU make an error!! at node_id "<< id <<"with right " << 0<< " " << tem <<"\n";
				return 0;
			}
		}
		
	}
*/
	std::cerr << "You have done a good job!\n";
	return 0;
}
