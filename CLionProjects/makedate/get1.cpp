//
// Created by root on 2020/5/26.
//

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
int main() {
    //freopen("/root/CLionProjects/test/data.txt","r",stdin);
    freopen("/root/CLionProjects/test/data0.txt","w",stdout);
    char a[10000];
    for(int i=1;i<=400;i++){
        scanf("%s",a+1);
        int len=0;
        for(int j=1;;j++){
            if(a[j]=='\0') break;
            len++;
        }
        if(a[len-7]=='0'){
            for(int j=1;j<=len-9;j++){
                printf("%c",a[j]);

            }
            printf("\n");
        }
    }
    return 0;
}
