#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdio.h>
float sigmoid(float x)
{
    return (1 / (1 + pow(2.71828182846f,x)));
}
int x[200];
float w[200];
char a[200000];
int main() {
    freopen("/home/hqf/CLionProjects/train.txt","r",stdin);
    freopen("/home/hqf/CLionProjects/train2.txt","w",stdin);
    while(scanf("%s",a+1)!=EOF){
        if(a[1]=='-'){
            printf("0 ");
        }
        else printf("%s",a+1);
    }
    return 0;
}
