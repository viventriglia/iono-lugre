#include <stdio.h>
#include <stdlib.h>
void setLibPath(const char*);
void GCPM(float*,float*,float*,int*,int*,float*,float*,int,float*,float*,float*,float*,int);
/* drv_sm libpath year doy ut kp < "x y z" in SM, Earth radii ; prints ne (cm^-3), like PyGCPM.GCPM */
int main(int c,char**v){ setLibPath(v[1]); int y=atoi(v[2]),d=atoi(v[3]); float ut=atof(v[4]),kp=atof(v[5]); float x,yy,z,ne,a,b,cc;
 while(scanf("%f %f %f",&x,&yy,&z)==3){ GCPM(&x,&yy,&z,&y,&d,&ut,&kp,1,&ne,&a,&b,&cc,0); printf("%.6g\n",ne);} return 0;}
