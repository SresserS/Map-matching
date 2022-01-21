#include<bits/stdc++.h>
#include<hash_map>
using namespace std;
template <class T>
inline bool scan_d(T &ret) {
    char c;
    int sgn;
    if (c = getchar(), c == EOF)
        return 0;  // EOF
    while (c != '-' && (c < '0' || c > '9')) c = getchar();
    sgn = (c == '-') ? -1 : 1;
    ret = (c == '-') ? 0 : (c - '0');
    while (c = getchar(), c >= '0' && c <= '9') ret = ret * 10 + (c - '0');
    ret *= sgn;
    return 1;
}


//vector change to array

class point{
public:
    int time;   
    double x,y;
};

class xypoint{
public:
    xypoint(){x=0,y=0;};
    xypoint(const double &x1,const double &y1);
    xypoint(const xypoint& p);
    double x,y;
};

class edge{
public:
    edge();
    //string way_str;
    int id,pointnum;
    double length;
    int startidx,endidx,way_type;
    vector<xypoint> unit;
};

edge::edge(){
    //string way_str="";
    id=pointnum=startidx=endidx=way_type=0;
    length=0;
    unit.clear();
}


class grid{
public:
    grid();
    void init();
    pair<int,int> nearEdges(const xypoint&p);
    double upperBoundry,lowerBoundry,leftBoundry,rightBoundry;
    double height=0.00855,width=0.00855;//单元的长和宽 约/800,/800  ///0.007可以
    int szH,szW;
    vector<int> units[2000][2000];
    int sz[2000][2000]={0};
    ////////
    //30.6406 31.4778
    //121.004 122.093
    /////////
    //确定网格的个数
    /////////
};

class Trajectory{
public:
    
    int idx;
    vector<point> traj;
};


xypoint::xypoint(const double &x1,const double &y1){
    x=x1;
    y=y1;
}
xypoint::xypoint(const xypoint& p){
    x=p.x;
    y=p.y;
}


///////////////////////////////////////////////
//全局变量声明区
struct node_{
    int edg;
    int to;
    double len;
};
vector<node_> node[65000];
int N,M;
vector<edge> edges;
vector<Trajectory> trajectories;
grid grids;
map<int,int> sameroad;
map<int,double> curprob;
map<int,int> lastedges;

//const double beta=5.1;//????????

const double pi=3.14;
const double lati=1.1;
const double atti=1.1;


///////////////////////////////////////////////

grid::grid(){}

void grid::init(){
    upperBoundry=rightBoundry=0.0;
    leftBoundry=lowerBoundry=200.0;

    for(int i=0;i<edges.size();i++){
        for(int j=0;j<edges[i].unit.size();j++){
            //upperBoundry=max(upperBoundry,edges[i].unit[j].y);
            lowerBoundry=min(lowerBoundry,edges[i].unit[j].y);
            //rightBoundry=max(rightBoundry,edges[i].unit[j].x);
            leftBoundry=min(leftBoundry,edges[i].unit[j].x);
        }
    }

    //szH=int(ceil((upperBoundry-lowerBoundry)/height));
    //szW=int(ceil((rightBoundry-leftBoundry)/width));

    for(int i=0;i<edges.size();i++){
        //map<int,int> vis;
        int prer,prec;
        for(int j=0;j<edges[i].unit.size();j++){
            int row=int(floor(lati*(edges[i].unit[j].y-lowerBoundry)/height));
            int col=int(floor(lati*(edges[i].unit[j].x-leftBoundry)/width));
            //去重啊...
            units[row][col].push_back(i);
            sz[row][col]++;
            
            /*if(j>0){
                //cout<<'('<<prer<<','<<prec<<','<<row<<','<<col<<')';
                //cout<<"   ";

                double k=(edges[i].unit[j].y-edges[i].unit[j-1].y)/(edges[i].unit[j].x-edges[i].unit[j-1].x);
                
                for(int p=prec+1;p<=col;p++){
                    double x=(p*width)/lati-edges[i].unit[j-1].x+leftBoundry;
                    int r=int(floor(lati*(k*x+edges[i].unit[j-1].y-lowerBoundry)/height));
                    if(sz[r][p]==0||units[r][p][sz[r][p]-1]!=i) units[r][p].push_back(i),sz[r][p]++;
                    //cout<<r<<' ';
                }

                k=1.0/k;
                for(int p=prer+1;p<row;p++){
                    double y=(p*height)/lati-edges[i].unit[j-1].y+lowerBoundry;
                    int c=int(floor(lati*(k*y+edges[i].unit[j-1].x-leftBoundry)/width));
                    if(sz[p][c]==0||units[p][c][sz[p][c]-1]!=i) units[p][c].push_back(i),sz[p][c]++;
                }

            }*/
            //cout<<endl;
            prer=row,prec=col;

        }
    }
}

//不用处理啊，直接返回网格的units就好，排除一些情况；
pair<int,int> grid::nearEdges(const xypoint &p){
    //vector<int> ans;
    
    //算到哪个网格里？
    int row=int(floor(lati*(p.y-lowerBoundry)/height));
    int col=int(floor(lati*(p.x-leftBoundry)/width));
    /*for(int i=0;i<units[row][col].size();i++){
        ans.push_back(units[row][col][i]);
    }*/
    return make_pair(row,col);
    /////
    //待施工，合理距离排除法
    /////
    //return units[row][col];
}


double dis_Between_2point(const xypoint& p1,const xypoint&p2){
    return sqrt(lati*(p1.x-p2.x)*lati*(p1.x-p2.x)+lati*(p1.y-p2.y)*lati*(p1.y-p2.y));
}

void filein(){
//Information of edges
    edge before;before.startidx=before.endidx=0;

    xypoint p;
    double x_c,y_c;
    string way_string;
    cin>>N;
    for(int j=0;j<N;j++){
        edge e;
        scan_d(e.id);
        scan_d(e.startidx);scan_d(e.endidx);
        cin>>way_string;///////////////////////////////
        scan_d(e.way_type);
        scan_d(e.pointnum);
        for(int i=0;i<e.pointnum;i++){
            scanf("%lf%lf",&x_c,&y_c);
            p.x=x_c,p.y=y_c;
            e.unit.emplace_back(p);
        }
        //e.length=dis_Between_2point(*e.unit.begin(),*(e.unit.end()-1));
        for(int i=0;i<e.unit.size()-1;i++){
            e.length+=dis_Between_2point(e.unit[i],e.unit[i+1]);
        }
        edges.emplace_back(e);
        node_ nn;nn.edg=e.id;nn.to=e.endidx,nn.len=e.length,nn.edg=e.id;
        node[e.startidx].push_back(nn);
        if(e.endidx==before.startidx&&e.startidx==before.endidx){
            sameroad[e.id]=before.id;
            sameroad[before.id]=e.id;
            //cout<<e.id<<' '<<before.id<<endl;
        }

        before.startidx=e.startidx,before.endidx=e.endidx,before.id=e.id;

    }

//Info of trajectory
    int ti;
    double x_m,y_m;
    scan_d(M);
    for(int i=0;i<M;i++){
        Trajectory tra;
        tra.idx=i;
        point poi;

        while(1){
            cin>>ti;/////////////////////////////////轨迹的时间不重要
            if(ti<10000) break;  ////////////////////
            //poi.time=ti;
            //cout<<poi.time<<' ';
            scanf("%lf%lf",&poi.x,&poi.y);
            //scan_d(poi.x);scan_d(poi.y);;
            tra.traj.emplace_back(poi);
            
        }
        trajectories.emplace_back(tra);
    }

}


double S_area(const xypoint& pt,const xypoint& egp1,const xypoint& egp2){
    double x1=(egp1.x-pt.x)*lati;
    double x2=(egp2.x-pt.x)*lati;
    double y1=(egp1.y-pt.y)*lati;
    double y2=(egp2.y-pt.y)*lati;
    return abs((x1*y2-x2*y1)/2);
}

bool Able_To_Cal_High(const xypoint&p1,const xypoint&p2,const xypoint&ori){
    double x1=p1.x-ori.x;//p1是pt，不变
    double x2=p2.x-ori.x;
    double y1=p1.y-ori.y;
    double y2=p2.y-ori.y;
    double ans1=x1*x2+y1*y2;

    x1=ori.x-p2.x;
    x2=p1.x-p2.x;
    y1=ori.y-p2.y;
    y2=p1.y-p2.y;
    double ans2=x1*x2+y1*y2;
    if(ans1>0&&ans2>0) return 1;
    else return 0;
}

double dist_poi_edge(const xypoint& pt,const edge& eg){
    double dist=1e9+7.0;
    double temp=0;
    double ans=0;
    for(int i=1;i<eg.unit.size();i++){
        //temp=dist_poi_2poi(pt,eg.unit[i-1],eg.unit[i]);
        temp=min(dis_Between_2point(pt,eg.unit[i-1]),dis_Between_2point(pt,eg.unit[i]));
        
        if(Able_To_Cal_High(pt,eg.unit[i-1],eg.unit[i])){
            double S=S_area(pt,eg.unit[i-1],eg.unit[i]);
            double hi=2.0*S/dis_Between_2point(eg.unit[i-1],eg.unit[i]);
            temp=min(temp,abs(hi));
            xypoint p1=eg.unit[i-1],p2=eg.unit[i];
            double x1=(pt.x-p1.x)*lati;
            double x2=(p2.x-p1.x)*lati;
            double y1=(pt.y-p1.y)*lati;
            double y2=(p2.y-p1.y)*lati;
            ans=x1*x2+y1*y2;
            //ans=abs(ans);
            ans/=dis_Between_2point(eg.unit[i-1],eg.unit[i]);
            ans/=dis_Between_2point(pt,p2);
        }
        dist=dist<temp?dist:temp;
        //cout<<dist<<' ';/////////////0.001级别
    }

    return dist;
}


//注意输入数据的顺序，仅仅考虑两条路
///注意输入顺序
double dist_start_point(const xypoint& pt,const edge &edg){
    double dist=0,pre=0;
    int sz=edg.unit.size();
    int i;
    for(i=0;i<sz-1;i++){
        //auto p1=edg.unit[i],p2=edg.unit[i+1];
        if(Able_To_Cal_High(pt,edg.unit[i],edg.unit[i+1])){
            break;
        }
        dist+=dis_Between_2point(edg.unit[i],edg.unit[i+1]);
    }
    xypoint p1=edg.unit[i],p2=edg.unit[i+1];
    double x1=(pt.x-p1.x)*lati;
    double x2=(p2.x-p1.x)*lati;
    double y1=(pt.y-p1.y)*lati;
    double y2=(p2.y-p1.y)*lati;
    double ans=x1*x2+y1*y2;
    ans=abs(ans);
    ans/=dis_Between_2point(edg.unit[i],edg.unit[i+1]);
    //ans=ans<dist?ans:dist;
    //cout<<ans<<' ';
    dist+=ans;
    //dist+=dis_Between_2point(edg.unit[i],edg.unit[i+1]);
    //cout<<dist<<" "; //有问题，一堆几百的负数·························
    return dist;
}

double Dijkstra(const edge&edg1,const edge&edg2){
    map<int,double> dist;
    map<int,int> v;
    priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> q;
    for(int i=0;i<60000;i++){
        dist[i]=10000;
    }

    q.push(make_pair(0,edg1.endidx));
    dist[edg1.endidx]=0;
    while(!q.empty()){
        pair<double,int> p=q.top();
        if(p.second==edg2.startidx) break;
        q.pop();
        int nx=p.second;
        if(dist[nx]<p.first){
            continue;
        }
        for(int i=0;i<node[nx].size();i++){
            node_ nn=node[nx][i];
            if(dist[nx]+nn.len<dist[nn.to]){
                dist[nn.to]=dist[nx]+nn.len;
                q.push(make_pair(dist[nn.to],nn.to));
            }
        }

    }
    //cout<<dist[edg2.startidx]<<' ';
    return dist[edg2.startidx];
}


vector<map<int,int>> paths;
struct redouble{double x=100.0;};
int lastedge[58005];
double distD[58005];
void dijk(){
    for(int j=0;j<58000;j++){
        distD[j]=100.0;
        //lastedge[j]=0;
    }
    //memset(lastedge,0,sizeof(lastedge));
    for(int i=0;i<58000;i++){
        map<int,int> path;
        map<int,int> remem;
        priority_queue<pair<double,int>,
        vector<pair<double,int>>,greater<pair<double,int>>> q;
        
        //cout<<lastedge[100]<<' ';
        distD[i]=0;
        lastedge[i]=0;
        pair<double,int> p;
        q.push(make_pair(0,i));
        remem[i]=1;
        while(!q.empty()){
            p=q.top(),q.pop();
            path[p.second]=lastedge[p.second];
            for(int j=0;j<node[p.second].size();j++){
                //cout<<dist[node[p.second][j].to]<<' ';
                if(distD[p.second]+node[p.second][j].len<distD[node[p.second][j].to]){
                    distD[node[p.second][j].to]=distD[p.second]+node[p.second][j].len;
                    lastedge[node[p.second][j].to]=node[p.second][j].edg;
                    q.push(make_pair(distD[node[p.second][j].to],node[p.second][j].to));
                    //cout<<distD[node[p.second][j].to]<<' ';
                    remem[node[p.second][j].to]=1;
                }
            }
            if(q.top().first>0.013||path.size()>50) break;
        }
        
        for(auto it=remem.begin();it!=remem.end();it++){
            distD[it->first]=100.0;
            //lastedge[it->first]=0;
        }
        //cout<<remem.size()<<' ';
        paths.push_back(path);
        
        ///////
        //把dist改变的点再设置回去
        //////
        
    }

}



struct edgenode{
    int edg_from;
    int edg;
    int ed;
    double w;
};
bool operator<(const edgenode&a,const edgenode&b){
    return a.w>b.w;
}

void dijk_prior(){
    for(int i=0;i<58000;i++){
        int vis[58005];
        map<int,int> path;
        //priority_queue<pair<double,int>,
        //vector<pair<double,int>>,greater<pair<double,int>>> q;
        priority_queue<edgenode> q;
        lastedge[i]=0;
        edgenode nn;
        nn.edg_from=0,nn.ed=i,nn.w=0,nn.edg=0;
        q.push(nn);
        while(!q.empty()){
            while(!q.empty()){
                nn=q.top();
                q.pop();
                if(!vis[nn.ed]) break;
            }
            path[nn.ed]=nn.edg_from;
            vis[nn.ed]=1;
            if(nn.w>0.03) break;
            for(int j=0;j<node[nn.ed].size();j++){
                edgenode nt;
                nt.w=node[nn.ed][j].len+nn.w;
                nt.ed=node[nn.ed][j].to;
                nt.edg_from=nn.edg;
                nt.edg=node[nn.ed][j].edg;
                q.push(nt);
            }
        }
        
        paths.push_back(path);
        
        ///////
        //把dist改变的点再设置回去
        //////
        
    }
}

double dijlen(const edge &edg1,const edge &edg2){
        map<int,int> vis;
        map<int,int> path;
        //map<int,int> remem;
        unordered_map<int,double> distD;
        priority_queue<pair<double,int>,
        vector<pair<double,int>>,greater<pair<double,int>>> q;
        
        //cout<<lastedge[100]<<' ';
        distD[edg1.endidx]=0;
        pair<double,int> p;
        q.push(make_pair(0,edg1.endidx));
        //remem[i]=1;
        while(!q.empty()){
            p=q.top(),q.pop();
            //remem[p.second]=1;
            if(p.second==edg2.startidx) return p.first;;
            if(p.first>0.02) return 10;
            for(int j=0;j<node[p.second].size();j++){
                //cout<<dist[node[p.second][j].to]<<' ';
                if(distD[p.second]==0&&p.second!=edg1.endidx) distD[p.second]=100;
                if(distD[node[p.second][j].to]==0) distD[node[p.second][j].to]=100;
                if(distD[p.second]+node[p.second][j].len<distD[node[p.second][j].to]){
                    distD[node[p.second][j].to]=distD[p.second]+node[p.second][j].len;
                    q.push(make_pair(distD[node[p.second][j].to],node[p.second][j].to));
                    //cout<<distD[node[p.second][j].to]<<' ';
                    //remem[node[p.second][j].to]=1;
                }
            }
            //if(q.top().first>0.0125) break;
            
        }
        return 10;

}

map<pair<int,int>,double> stmp;
double ST_distance(const edge&edg1,const edge&edg2){
    map<int,int> &mp=paths[edg1.endidx];
    //cout<<mp.size()<<' ';
    //////////////注意找的是E2的最后的节点的
    /*if(!mp[edg2.startidx]){
        return 0;
    }*/
    if(stmp[make_pair(edg1.endidx,edg2.startidx)]) return stmp[make_pair(edg1.endidx,edg2.startidx)];
    if(!mp[edg2.startidx]){
        stmp[make_pair(edg1.endidx,edg2.startidx)]=dijlen(edg1,edg2);
        return stmp[make_pair(edg1.endidx,edg2.startidx)];
    }
    
    int ed=edg2.startidx;
    vector<int> pa;


    for(;mp[ed]&&ed!=edg1.endidx;ed=edges[mp[ed]].startidx){
        pa.push_back(mp[ed]);
    }

    double dist=0;
    for(int i=0;i<pa.size();i++){
        dist+=edges[pa[i]].length;
    }

    return dist;

}


double dist_ri_rj(const xypoint&p1,const xypoint&p2,const edge&edg1,const edge&edg2){
    double dist_road=0;
    double r1,r2;
    r1=dist_start_point(p1,edg1);
    r2=dist_start_point(p2,edg2);
    if(edg1.id==edg2.id||edg2.id==sameroad[edg1.id]){
        dist_road=abs(r1-r2);
    }   
    /*else if(edg1.startidx==edg2.startidx){
        double len1=dis_Between_2point(*edg1.unit.begin(),*(edg1.unit.end()-1));
        double len2=dis_Between_2point(*edg2.unit.begin(),*(edg2.unit.end()-1));
        dist_road=r1+r2;
    }
    else if(edg1.endidx==edg2.endidx){
        double len1=dis_Between_2point(*edg1.unit.begin(),*(edg1.unit.end()-1));
        double len2=dis_Between_2point(*edg2.unit.begin(),*(edg2.unit.end()-1));
        dist_road=len1+len2-r1-r2;
    }
    else if(edg1.startidx==edg2.endidx){
        double len1=dis_Between_2point(*edg1.unit.begin(),*(edg1.unit.end()-1));
        double len2=dis_Between_2point(*edg2.unit.begin(),*(edg2.unit.end()-1));
        dist_road=r1+len2-r2;
    }*/
    else{
        //百米的量级
        /*double len1=dis_Between_2point(*edg.unit.begin(),*(edg1.unit.end()-1));
        double len2=dis_Between_2point(*edg2.unit.begin(),*(edg2.unit.end()-1));*/
        double len1=edg1.length;
        dist_road=len1-r1+r2+ST_distance(edg1,edg2);////////////之前为什么加了len2？
    }
    //cout<<dist_road<<" ";
    //简单修改法：求另一种方法，比较哪个最接近
    return abs(dist_road);
}

//sqrt(2*pi)*sig=18.08
const double sig=0.00807;//?????
double visualPosb(const xypoint&p,const xypoint &pt,const edge& r){
    double exnum,dist,ans;

    double level=double(r.way_type);
    level=-level*0.08+1;//0.08，0.15
    double cos=1.0;
    int i=0;
    /*for(i=0;i<r.unit.size()-1;i++){
        if(Able_To_Cal_High(p,r.unit[i],r.unit[i+1]));
    }

    if(pt.x!=0){
        cos=(p.x-pt.x)*(r.unit[i+1].x-r.unit[i].x)+(p.y-pt.y)*(r.unit[i+1].y-r.unit[i].y);
        cos=cos/dis_Between_2point(r.unit[i],r.unit[i+1])/dis_Between_2point(p,pt);
    }*/
    //cout<<cos<<' ';
    dist=dist_poi_edge(p,r);

    for(i=0;i<r.unit.size()-1;i++){
        if(Able_To_Cal_High(p,r.unit[i],r.unit[i+1])) break;
    }

    if(i==r.unit.size()-1){
        dist=dist*5;    
    }

    
    dist=dist*1.2;

    dist=level*dist;
    //dist=dist*(1-0.05*cos);
    //if(dist>5.0*1e-3) return 0;////////5e-3还是75分
    //cout<<dist<<" ";
    exnum=-(dist/sig)*(dist/sig)/2;
    ans=exp(exnum)/(0.05);
    
    //ans=ans*level;
    //cout<<ans<<' '; 
    return ans;
}

const double beta[31]={5,0.49,0.82,1.24,1.67,2.00,2.42,//5
2.81,3.15,3.52,4.10,4.67,5.41,
6.47,6.29,7.80,8.09,8.08,9.09,
11.09,11.88,12.55,15.83,17.69,18.07,
19.63,25.4,23.75,28.43,32.22,34.57};


double stateTransPosb(const xypoint&p1,const xypoint&p2,const edge&e1,const edge&e2,const int &t1,const int &t2){
    double dist;
    dist=dist_ri_rj(p1,p2,e1,e2);
    /*else{
        dist=dist_ri_rj(p1,p2,e1,e2);
        
    }*/

    dist=abs(dis_Between_2point(p1,p2)-dist);
    //if(e1.way_type-e2.way_type>=3) dist*=30;
    //if(dist>5.0*1e-2) return 0;///////////////////大于这个都是75分没变。。
    //cout<<dist<<' ';
    if(t2-t1<=30){
        double m=10.0*dist/beta[t2-t1];///////////1.0-100.0都是71分
        //cout<<exp(-m)<<' ';
        return exp(-m)/beta[t2-t1];
    }
    else {
        double m=10.0*dist/35.0;///////////1.0-100.0都是71分
    //cout<<exp(-m)<<' ';
        return exp(-m)/35.0;
    }
}


bool roadconnect(const edge&e1,const edge&e2){
    if(e1.startidx==e2.startidx||e1.startidx==e2.endidx||e1.endidx==e2.startidx||e1.endidx==e2.endidx){
        return true;
    }
    else return false;
}

double V(const point&p1,const point&p2){
    double speed=0,dist;
    xypoint pt1,pt2;
    pt1.x=p1.x,pt1.y=p1.y;
    pt2.x=p2.x,pt2.y=p2.y;
    dist=dis_Between_2point(pt1,pt2);
    if(p2.time-p1.time!=0)speed=dist/(p2.time-p1.time);
    return 360000*speed;
}

const double roadspeed[8]{
    0,10,20,35,50,75,90,110
};



void match(const Trajectory&traj,vector<map<int,int>> &matchedEdgs,
    map<int,double> &hiddenProb){
    point pos;
    xypoint p,pt;
    double visprob;
    int t1,t2;
    //vector<int> matched;
    for(int i=0;i<traj.traj.size();i++){
        pos=traj.traj[i];
        p.x=pos.x,p.y=pos.y;
        t1=pos.time;
        /////////////
        //改数组，不会越界
        /////////////
        
        curprob.clear();
        lastedges.clear();
        double mxpro=-1.0,sum=0.0,transpro,lastedge;
        pair<int,int> rowcol;
        rowcol=grids.nearEdges(p);/////确定合理距离//////
        vector<int> &gridEdgs=grids.units[rowcol.first][rowcol.second];
        priority_queue<pair<double,int>> q;
        
        if(i>0&&rowcol.first<2000&&rowcol.second<20000&&rowcol.first>=0&&rowcol.second>=0){
            map<int,double> visprobs;
            for(int idx=0;idx<gridEdgs.size();idx++){
                visprob=visualPosb(p,pt,edges[gridEdgs[idx]]);
                visprobs[gridEdgs[idx]]=visprob;//////////////////////
            }
            pt.x=traj.traj[i-1].x,
            pt.y=traj.traj[i-1].y;
            t2=traj.traj[i-1].time;
            auto &lastprob=hiddenProb;////////???

            //double speed=V(traj.traj[i-1],traj.traj[i]);
            //cout<<speed<<endl;;
            
            for(int idx=0;idx<gridEdgs.size();idx++){
                mxpro=-1.0;//////////////////////////////////////-1
                double sepro=-1;
                int seedge=0;
                for(auto it=lastprob.begin();it!=lastprob.end();it++){
                    transpro=stateTransPosb(pt,p,edges[it->first],edges[gridEdgs[idx]],t2,t1);
                    //cout<<it->second<<' ';
                    sum=(it->second)*transpro;////////it->second 为0 ?X

                    //double standard=roadspeed[edges[gridEdgs[idx]].way_type];
                    //cout<<standard<<' ';
                    //sum=sum*(1-abs(standard-speed)/200.0);
                    //cout<<speed<<' ';

                    if(sum>mxpro){//&&roadconnect(edges[it->first],edges[gridEdgs[idx]])
                        sepro=mxpro,seedge=lastedge;
                        mxpro=sum,lastedge=it->first;
                        
                    }
                    if(sum!=mxpro&&sum>sepro){
                        sepro=sum,seedge=it->first;
                    }
                    

                }

                
                /*for(auto it=lastprob.begin();it!=lastprob.end();it++){
                    transpro=stateTransPosb(pt,p,edges[it->first],edges[gridEdgs[idx]],0,0);

                    sum=(it->second)*transpro;
                    if(sum!=mxpro&&sum>sepro){
                        sepro=sum,seedge=it->first;
                    }
                }*/
                
                if(sameroad[lastedge]){
                    double x1,y1,x2,y2;
                    int sz=edges[lastedge].unit.size();
                    x1=p.x-pt.x,y1=p.y-pt.y;
                    x2=edges[lastedge].unit[sz-1].x-edges[lastedge].unit[0].x;
                    y1=p.y-pt.y;
                    y2=edges[lastedge].unit[sz-1].y-edges[lastedge].unit[0].y;
                    
                    double judge=x1*x2+y1*y2;
                    if(judge<0) lastedge=sameroad[lastedge];
                    //cout<<judge<<' ';
                }

                /*else if(seedge!=0){
                    int sz=edges[lastedge].unit.size();
                    int sz2=edges[seedge].unit.size();
                    double cos1=(p.x-pt.x)*(edges[lastedge].unit[sz-1].x-edges[lastedge].unit[0].x)+
                                (p.y-pt.y)*(edges[lastedge].unit[sz-1].y-edges[lastedge].unit[0].y);
                    double cos2=(p.x-pt.x)*(edges[seedge].unit[sz2-1].x-edges[seedge].unit[0].x)+
                                (p.y-pt.y)*(edges[seedge].unit[sz2-1].y-edges[seedge].unit[0].y);
                    double di=dis_Between_2point(p,pt);
                    if(cos1<0&&cos2>0&&di>0.01){
                        lastedge=seedge;
                        mxpro=sepro;
                    } 
                }*/


                /*if(sameroad[lastedge]){
                    int j=0;
                    double cos=0;
                    for(j=0;j<gridEdgs.size()-1;j++){
                        if(Able_To_Cal_High(p,edges[lastedge].unit[j],edges[lastedge].unit[j+1])) break;
                    }

                    double x1,y1,x2,y2;
                    int sz=edges[lastedge].unit.size();
                    x1=p.x-pt.x,y1=p.y-pt.y;
                    x2=edges[lastedge].unit[j+1].x-edges[lastedge].unit[0].x;
                    y1=p.y-pt.y;
                    y2=edges[lastedge].unit[j+1].y-edges[lastedge].unit[0].y;

                    double judge=x1*x2+y1*y2;
                    if(judge<0) lastedge=sameroad[lastedge];
                    //cout<<judge<<' ';
                }*/

                lastedges[gridEdgs[idx]]=lastedge;
                //curprob[gridEdgs[idx]]=mxpro;
                curprob[gridEdgs[idx]]=visprobs[gridEdgs[idx]]*mxpro; //////////////乘法改加法
                    ////////////////
                    //修改为先一起算上vis概率
                    ////////////////
                
                //cout<<curprob[gridEdgs[idx]]<<' ';
            }
            double mother=0;
            for(auto jt=curprob.begin();jt!=curprob.end();jt++){
                mother=max(mother,jt->second);
            }
            for(auto jt=curprob.begin();jt!=curprob.end();jt++){
                jt->second/=mother;
            }
            
           hiddenProb=curprob;

        }
        else{
            map<int,double> visprobs;
            for(int idx=0;idx<gridEdgs.size();idx++){
                visprob=visualPosb(p,pt,edges[gridEdgs[idx]]);
                visprobs[gridEdgs[idx]]=visprob;//////////////////////
            }
            double mother=0;
            for(auto jt=visprobs.begin();jt!=visprobs.end();jt++){
                mother=max(mother,jt->second);
            }
            for(auto jt=visprobs.begin();jt!=visprobs.end();jt++){
                jt->second/=mother;
            }
            //curprob=visprobs;

            hiddenProb=visprobs;
        }
        matchedEdgs.push_back(lastedges);
        //hiddenProb.push_back(curprob);
    }

    //double mx=-1;
    /*int edg;
    
    //cout<<edg<<' ';
    ////////////////////////////////////////////////////double太小不能比大小。。。。放大。。。。。
    

    /*pt.x=traj.traj[0].x,pt.y=traj.traj[0].y;
    p.x=traj.traj[1].x,pt.y=traj.traj[1].y;
    if(sameroad[matched[0]]){
        double x1,y1,x2,y2;
        int sz=edges[matched[0]].unit.size();
        x1=p.x-pt.x,y1=p.y-pt.y;
        x2=edges[matched[0]].unit[sz-1].x-edges[matched[0]].unit[0].x;
        y1=p.y-pt.y;
        y2=edges[matched[0]].unit[sz-1].y-edges[matched[0]].unit[0].y;

        double judge=x1*x2+y1*y2;
        if(judge<0) matched[0]=sameroad[matched[0]];        
    }*/

    //给0项最近点匹配


    /*int flag=0;
    if(mxp-secp<0.05&&mxp&&secp){
        flag=1;
        if(pidx!=0&&matched[pidx-1]!=matched[pidx]){
            if(edges[matched[pidx-1]].endidx!=edges[matched[pidx]].startidx
            &&edges[matched[pidx-1]].endidx==edges[gridEdgs[sidx]].startidx){
                flag=1;
            }
            
        }

    }

    for(int i=pidx;flag&&(i>=pidx-num);i-=2){
        matched[i]=gridEdgs[sidx];
    }*/

    //int sz=matched.size();
    /*pt.x=traj.traj[0].x,pt.y=traj.traj[0].y;
    p.x=traj.traj[1].x,pt.y=traj.traj[1].y;
    if(sameroad[matched[0]]){
        double x1,y1,x2,y2;
        int sz=edges[matched[0]].unit.size();
        x1=p.x-pt.x,y1=p.y-pt.y;
        x2=edges[matched[0]].unit[sz-1].x-edges[matched[0]].unit[0].x;
        y1=p.y-pt.y;
        y2=edges[matched[0]].unit[sz-1].y-edges[matched[0]].unit[0].y;

        double judge=x1*x2+y1*y2;
        if(judge<0) matched[0]=sameroad[matched[0]];        
    }


    for(int i=0;i<sizeofedg;i++){
       printf("%d ",matched[i]);
    }
    printf("\n");*/

}

void ansofmm(Trajectory&traj,vector<map<int,int>> &matchedEdgs,
    map<int,double> &hiddenProb){
    int edg;

    double mx=-1;
    for(auto it=hiddenProb.begin();it!=hiddenProb.end();it++){
        if((it->second) > mx){
            mx=it->second;
            edg=it->first;
        }
        //cout<<it->second<<" ";
    }
    const int sizeofedg=matchedEdgs.size();
    int matched[sizeofedg];
    matched[sizeofedg-1]=edg;
    for(int i=matchedEdgs.size()-1;i>0;i--){
        edg=matchedEdgs[i][edg];        
        //matched.push_back(edg);
        matched[i-1]=edg;
    }
    xypoint pt,p;
    pt.x=traj.traj[0].x,pt.y=traj.traj[0].y;
    p.x=traj.traj[1].x,pt.y=traj.traj[1].y;

    for(int i=0;i<sizeofedg;i++){
        for(int j=i+2;j<sizeofedg;j++){//+2 93.45
            if(matched[i]==matched[j]){
                for(int k=i+1;k<j;k++){
                    matched[k]=matched[i];
                }
            }
        }
    }


    /*if(sameroad[matched[0]]){
        double x1,y1,x2,y2;
        int sz=edges[matched[0]].unit.size();
        x1=p.x-pt.x,y1=p.y-pt.y;
        x2=edges[matched[0]].unit[sz-1].x-edges[matched[0]].unit[0].x;
        y1=p.y-pt.y;
        y2=edges[matched[0]].unit[sz-1].y-edges[matched[0]].unit[0].y;

        double judge=x1*x2+y1*y2;
        if(judge<0) matched[0]=sameroad[matched[0]];        
    }*/

    /*int pidx=0,num=0,flag=0;
    double visprob,prex=0,prey=0;
    pair<int,int> rowcol;
    xypoint poi;
    map<int,double> visprobs;
    
    for(int i=0;i<traj.traj.size();i++){
        if(prex==traj.traj[i].x&&prey==traj.traj[i].y){
            pidx=i,num++;
        }
        if(num!=0&&(prex!=traj.traj[i].x||prey!=traj.traj[i].y)) break;
        prex=traj.traj[i].x,prey=traj.traj[i].y;
    }
    
    poi.x=traj.traj[pidx].x,poi.y=traj.traj[pidx].y;    
    edg=matchedEdgs[pidx][matched[pidx]];
    for(int i=0;i<edges[edg].unit.size()-1;i++){
        if(Able_To_Cal_High(poi,edges[edg].unit[i],edges[edg].unit[i+1])){
            flag=1;
        }
    }

    double mxp=0;
    if(flag==0){
        rowcol=grids.nearEdges(poi);/////确定合理距离//////
        vector<int> &gridEdgs=grids.units[rowcol.first][rowcol.second];
        double mxp=0,secp=0,midx=0,sidx=0;
        for(int idx=0;idx<gridEdgs.size();idx++){
            visprob=visualPosb(p,edges[gridEdgs[idx]]);
            visprobs[gridEdgs[idx]]=visprob;//////////////////////
        }
        auto jt=visprobs.find(visprobs[edg]);
        visprobs.erase(jt);
        for(auto it=visprobs.begin();it!=visprobs.end();it++){
            if(it->second>mxp){
                mxp=it->second;
                edg=it->first;
            }
        }
        if(edg!=0){
            for(int i=pidx;i>pidx-num;i--){
                matched[i]=edg;
            }
        }

    }*/
    
    


    for(int i=0;i<sizeofedg;i++){
       printf("%d ",matched[i]);
    }
    printf("\n");

}

void easymatch(const Trajectory &traj){
    for(int i=0;i<traj.traj.size();i++){
        xypoint p;
        p.x=traj.traj[i].x,p.y=traj.traj[i].y;
        pair<int,int> rc;
        rc=grids.nearEdges(p);
        vector<int> &temp=grids.units[rc.first][rc.second];
        double mx=1000000;
        int ans=0;
        for(int j=0;rc.first<2000&&rc.second<20000&&rc.first>=0&&rc.second>=0&&j<grids.sz[rc.first][rc.second];j++){
            double dist=dist_poi_edge(p,edges[temp[j]]);
            if(dist<mx) ans=edges[temp[j]].id,mx=dist;
        }
        cout<<ans<<' ';

    }
    cout<<endl;
}


void cmp(){
    
}


int main(){
   //freopen("sample.in","r",stdin);
   //freopen("co.out","w",stdout);
    clock_t startTime,endTime;
    //startTime = clock();//计时开始
    filein();
    //dijk();
    grids.init();
    dijk();
    cout<<M<<endl;
    //cout<<edges[22].way_type<<endl<<edges[93956].way_type;
    for(int i=0;i<trajectories.size();i++){
        vector<map<int,int>> matchedEdgs; 
        map<int,double> hiddenProb;
        match(trajectories[i],matchedEdgs,hiddenProb);
        ansofmm(trajectories[i],matchedEdgs,hiddenProb);
    }

    /*xypoint poi;
    poi.x=trajectories[3].traj[5].x,poi.y=trajectories[3].traj[5].y;
    double vis8224=visualPosb(poi,edges[8224]);
    double vis8247=visualPosb(poi,edges[8247]);
    cout<<vis8224<<endl<<vis8247;*/
    //19.9552,19.9804
    //cout<<num;
    //endTime=clock();
    //printf("%lf",edges[0].unit[0].x);
    //cout << "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

}





/////////////////////////////   


/////////////////////////////



