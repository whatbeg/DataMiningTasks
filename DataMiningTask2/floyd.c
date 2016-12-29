#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double graph[3201][3201];
char buf[500002];

int main()
{
    FILE *fpin = fopen("graph.txt", "r");
    char tmp[500012];
    char *p;
    const char * split = ",";
    int i = 0, n = 0, j = 0, k = 0;
    while(fgets(tmp, 500012, fpin)) {
        tmp[strlen(tmp)] = '\0';
        j = i;   //ÉÏÈý½Ç
        p = strtok(tmp, split);
        while(p != NULL) {
            graph[i][j++] = atof(p);
            graph[j-1][i] = graph[i][j-1];
            p = strtok(NULL, split);
        }
        i++;
        memset(tmp, 0, sizeof(tmp));
    }
    n = j-1;
    for(k=0;k<n;k++) {
        //printf("k = %d\n", k);
        for(i=0;i<n;i++) {
            for(j=0;j<n;j++) {
                if (graph[i][j] > graph[i][k] + graph[k][j]) {
                    //printf("HITS %f %f %f\n", graph[i][j], graph[i][k], graph[k][j]);
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }
    FILE *fpout = fopen("graph_floyded.txt", "w");
    for(i=0;i<n;i++) {
        for(j=i;j<n;j++) {
            //printf("%f\n", graph[i][j]);
            fprintf(fpout, "%.7f,", graph[i][j]);
        }
        fputs("\n", fpout);
    }
    fclose(fpin);
    fclose(fpout);
    return 0;
}
