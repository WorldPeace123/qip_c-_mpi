//
// 测试文件: test_qip.cpp
// 用于验证QIP量子内积计算的正确性
//

#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
using namespace std;

// 复数类型定义
struct C {
    double r, i;
    C(double r_=0.0, double i_=0.0): r(r_), i(i_) {}
};

static inline C cadd(const C&a, const C&b){ return C(a.r+b.r, a.i+b.i);} 
static inline C csub(const C&a, const C&b){ return C(a.r-b.r, a.i-b.i);} 
static inline C cmul(const C&a, const C&b){ return C(a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r);} 

// 2x2复数矩阵
struct M2 { C a00,a01,a10,a11; };

static inline M2 matI(){ return {C(1,0),C(0,0),C(0,0),C(1,0)}; }
static inline M2 matX(){ return {C(0,0),C(1,0),C(1,0),C(0,0)}; }
static inline M2 matZ(){ return {C(1,0),C(0,0),C(0,0),C(-1,0)}; }
static inline M2 matH(){ double s = 1.0/sqrt(2.0); return {C(s,0),C(s,0),C(s,0),C(-s,0)}; }
static inline M2 matRY(double th){ double c=cos(th/2.0), s=sin(th/2.0); return {C(c,0),C(-s,0),C(s,0),C(c,0)}; }

// 量子态模拟器
struct State {
    static constexpr int n = 9; // qubits
    static constexpr size_t Dim = (1u<<n);
    array<C, Dim> amp; // size 512
    State(){ amp.fill(C(0,0)); amp[0]=C(1,0); }

    // 单量子比特门
    void apply1(const M2&U, int q){
        const size_t N = Dim;
        const size_t step = 1ull<<q;
        const size_t period = step<<1;
        for(size_t base=0; base<N; base+=period){
            for(size_t k=0;k<step;++k){
                size_t i0 = base + k;
                size_t i1 = i0 + step;
                C a0 = amp[i0];
                C a1 = amp[i1];
                C b0 = cadd(cmul(U.a00,a0), cmul(U.a01,a1));
                C b1 = cadd(cmul(U.a10,a0), cmul(U.a11,a1));
                amp[i0]=b0; amp[i1]=b1;
            }
        }
    }

    // 多控制单量子比特门
    void applyCtrl1(const M2&U, const vector<int>& ctrls, int tgt){
        const size_t N = Dim;
        size_t mask=0; for(int c:ctrls) mask |= (1ull<<c);
        size_t tbit = 1ull<<tgt;
        for(size_t i=0;i<N;i++){
            if( ( i & mask) == mask ){
                size_t j = i ^ tbit;
                if( ((size_t)i & tbit)==0 ){
                    C a0 = amp[i];
                    C a1 = amp[j];
                    C b0 = cadd(cmul(U.a00,a0), cmul(U.a01,a1));
                    C b1 = cadd(cmul(U.a10,a0), cmul(U.a11,a1));
                    amp[i]=b0; amp[j]=b1;
                }
            }
        }
    }

    // 多控制双量子比特门
    void applyCtrl2(const array<C,16>& U4, const vector<int>& ctrls, int q1, int q2){
        const size_t N = Dim;
        size_t mask=0; for(int c:ctrls) mask |= (1ull<<c);
        int hi = max(q1,q2), lo=min(q1,q2);
        size_t bhi = 1ull<<hi, blo=1ull<<lo;
        for(size_t idx=0; idx<N; ++idx){
            if( (idx & mask) != mask ) continue;
            if( (idx & bhi) || (idx & blo) ) continue;
            size_t i00 = idx;
            size_t i01 = idx | blo;
            size_t i10 = idx | bhi;
            size_t i11 = idx | bhi | blo;
            C v00=amp[i00], v01=amp[i01], v10=amp[i10], v11=amp[i11];
            C in[4]={v00,v01,v10,v11};
            C out[4]={};
            for(int r=0;r<4;++r){
                C acc; 
                for(int c=0;c<4;++c){
                    const C &m = U4[r*4+c];
                    acc = cadd(acc, cmul(m, in[c]));
                }
                out[r]=acc;
            }
            amp[i00]=out[0]; amp[i01]=out[1]; amp[i10]=out[2]; amp[i11]=out[3];
        }
    }
};

// 层次归一化函数
static void hierarchical_normalization(const array<double,16>& a, array<double,15>& out){
    array<double,16> layer = a;
    int len = 16;
    int writePos = 0;
    while(len > 1){
        int nextLen = len/2;
        array<double,16> next{};
        for(int i=0;i<nextLen;++i){
            double x = layer[2*i], y = layer[2*i+1];
            next[i] = sqrt(x*x + y*y);
        }
        for(int i=nextLen-1;i>=0;--i){
            out[writePos++] = next[i];
        }
        for(int i=0;i<nextLen;++i) layer[i] = next[i];
        len = nextLen;
    }
}

// 量子电路白盒计算
static double Quantum_circuit_whitebox(const array<double,15>& theta_a, 
                                       const array<double,15>& theta_b,
                                       const array<int,16>& flag_a, 
                                       const array<int,16>& flag_b){
    State st;
    auto X=matX(), Z=matZ(), H=matH();
    auto RY=[&](double th){ return matRY(th); };

    auto CRY = [&](int c, int t, double th){ st.applyCtrl1(RY(th), {c}, t); };
    auto MCRY = [&](vector<int> ctrls, int t, double th){ st.applyCtrl1(RY(th), ctrls, t); };
    auto MCZ = [&](vector<int> ctrls, int t){ st.applyCtrl1(Z, ctrls, t); };
    auto MCX = [&](vector<int> ctrls, int t){ st.applyCtrl1(X, ctrls, t); };
    auto s1 = [&](int q, const M2&U){ st.apply1(U,q); };
    auto X1 = [&](int q){ s1(q,X);} ;
    auto H1 = [&](int q){ s1(q,H);} ;

    // CSWAP via controlled 2-qubit 4x4
    array<C,16> SWAP = { C(1,0),C(0,0),C(0,0),C(0,0),
                         C(0,0),C(0,0),C(1,0),C(0,0),
                         C(0,0),C(1,0),C(0,0),C(0,0),
                         C(0,0),C(0,0),C(0,0),C(1,0)};
    auto CSW = [&](int c, int a, int b){ st.applyCtrl2(SWAP, {c}, max(a,b), min(a,b)); };

    // 量子电路实现
    s1(1, RY(theta_a[0]));
    s1(5, RY(theta_b[0]));
    
    X1(1); X1(5);
    CRY(1,2, theta_a[1]);
    CRY(5,6, theta_b[1]);
    X1(1); X1(5);
    CRY(1,2, theta_a[2]);
    CRY(5,6, theta_b[2]);
    
    X1(1); X1(2);
    X1(5); X1(6);
    MCRY({1,2}, 3, theta_a[3]);
    MCRY({5,6}, 7, theta_b[3]);
    X1(1); X1(2);
    X1(5); X1(6);

    X1(1); X1(5);
    MCRY({1,2}, 3, theta_a[4]);
    MCRY({5,6}, 7, theta_b[4]);
    X1(1); X1(5);

    X1(2); X1(6);
    MCRY({1,2}, 3, theta_a[5]);
    MCRY({5,6}, 7, theta_b[5]);
    X1(2); X1(6);

    MCRY({1,2}, 3, theta_a[6]);
    MCRY({5,6}, 7, theta_b[6]);

    // 块处理
    auto block = [&](int ax, int ay, int bx, int by, double tha, double thb){
        MCRY({1,2,3}, 4, tha);
        if( (ax==1 && ay==-1) || (ax==0 && ay==-1) ) MCZ({1,2,3},4);
        if( ax==-1 && ay==1 ){ MCX({1,2,3},4); MCZ({1,2,3},4); MCX({1,2,3},4);} 
        if( (ax==-1 && ay==-1) || (ax==-1 && ay==0) ){ MCZ({1,2,3},4); MCX({1,2,3},4); MCZ({1,2,3},4); MCX({1,2,3},4);} 
        MCRY({5,6,7}, 8, thb);
        if( (bx==1 && by==-1) || (bx==0 && by==-1) ) MCZ({5,6,7},8);
        if( bx==-1 && by==1 ){ MCX({5,6,7},8); MCZ({5,6,7},8); MCX({5,6,7},8);} 
        if( (bx==-1 && by==-1) || (bx==-1 && by==0) ){ MCZ({5,6,7},8); MCX({5,6,7},8); MCZ({5,6,7},8); MCX({5,6,7},8);} 
    };

    // 执行所有块
    X1(1); X1(2); X1(3); X1(5); X1(6); X1(7);
    block(flag_a[0],flag_a[1],flag_b[0],flag_b[1], theta_a[7], theta_b[7]);
    X1(1); X1(2); X1(3); X1(5); X1(6); X1(7);
    
    X1(1); X1(2); X1(5); X1(6);
    block(flag_a[2],flag_a[3],flag_b[2],flag_b[3], theta_a[8], theta_b[8]);
    X1(1); X1(2); X1(5); X1(6);
    
    X1(1); X1(3); X1(5); X1(7);
    block(flag_a[4],flag_a[5],flag_b[4],flag_b[5], theta_a[9], theta_b[9]);
    X1(1); X1(3); X1(5); X1(7);
    
    X1(1); X1(5);
    block(flag_a[6],flag_a[7],flag_b[6],flag_b[7], theta_a[10], theta_b[10]);
    X1(1); X1(5);
    
    X1(2); X1(3); X1(6); X1(7);
    block(flag_a[8],flag_a[9],flag_b[8],flag_b[9], theta_a[11], theta_b[11]);
    X1(2); X1(3); X1(6); X1(7);
    
    X1(2); X1(6);
    block(flag_a[10],flag_a[11],flag_b[10],flag_b[11], theta_a[12], theta_b[12]);
    X1(2); X1(6);
    
    X1(3); X1(7);
    block(flag_a[12],flag_a[13],flag_b[12],flag_b[13], theta_a[13], theta_b[13]);
    X1(3); X1(7);
    
    block(flag_a[14],flag_a[15],flag_b[14],flag_b[15], theta_a[14], theta_b[14]);

    // Swap test
    H1(0);
    CSW(0,1,5); CSW(0,2,6); CSW(0,3,7); CSW(0,4,8);
    H1(0);

    // 计算概率
    double p0=0.0; const size_t N = State::Dim;
    for(size_t idx=0; idx<N; ++idx){
        if( (idx & 1ull)==0 ){
            p0 += st.amp[idx].r*st.amp[idx].r + st.amp[idx].i*st.amp[idx].i;
        }
    }
    return p0;
}

// 量子内积计算核心函数
static double q_inner(const array<double,16>& a_in, const array<double,16>& b_in){
    array<int,16> flag_a{}, flag_b{};
    for(size_t i=0;i<16;++i) flag_a[i] = (a_in[i]>0)?1:(a_in[i]==0?0:-1);
    for(size_t i=0;i<16;++i) flag_b[i] = (b_in[i]>0)?1:(b_in[i]==0?0:-1);
    array<double,16> a{}, b{};
    for(size_t i=0;i<16;++i){ a[i]=fabs(a_in[i]); b[i]=fabs(b_in[i]); }
    double norma = 0, normb = 0;
    for(int i=0;i<16;++i){ norma += a[i]*a[i]; normb += b[i]*b[i]; }
    norma = sqrt(norma); normb = sqrt(normb);
    if(norma==0 || normb==0) return 0.0;
    size_t index_a=0,index_b=0; double ka0=0,kb0=0;
    for(size_t i=0;i<16;++i){ if(a[i]!=0){ index_a=i; ka0=a[i]; break; } }
    for(size_t i=0;i<16;++i){ if(b[i]!=0){ index_b=i; kb0=b[i]; break; } }
    for(int i=0;i<16;++i){ a[i]/=norma; b[i]/=normb; }
    double k_a = a[index_a]/ka0;
    double k_b = b[index_b]/kb0;

    auto build_theta = [](const array<double,16>& vec){
        array<double,15> temp{};
        hierarchical_normalization(vec, temp);
        array<double,15> rtemp{};
        for(int i=0;i<15;++i) rtemp[i] = temp[14 - i];
        double concat[31];
        for(int i=0;i<15;++i) concat[i]=rtemp[i];
        for(int i=0;i<16;++i) concat[15+i]=vec[i];
        double res[31];
        res[0]=concat[0];
        for(int i=1;i<31;++i){ double parent = concat[(i-1)/2]; res[i] = (parent==0.0)?0.0:(concat[i]/parent); }
        array<double,15> theta{};
        int t=0; for(int i=1;i<31; i+=2){ double x=res[i]; if(x>1)x=1; if(x<-1)x=-1; theta[t++] = 2.0*acos(x); }
        return theta;
    };

    array<double,15> theta_a = build_theta(a);
    array<double,15> theta_b = build_theta(b);

    double p0 = Quantum_circuit_whitebox(theta_a, theta_b, flag_a, flag_b);
    if(p0 < 0.5) return 0.0; else return sqrt(max(0.0, 2.0*p0 - 1.0))/(k_a*k_b);
}

int main(int argc, char* argv[]) {
    cout << "QIP量子内积计算测试程序" << endl;
    cout << "========================" << endl;
    
    // 功能选项：选择计算类型
    int calculation_type;
    cout << "请选择计算类型:" << endl;
    cout << "1 - 向量乘向量" << endl;
    cout << "2 - 矩阵乘矩阵" << endl;
    cout << "3 - 向量乘矩阵" << endl;
    cout << "请输入选择 (1, 2 或 3): ";
    cin >> calculation_type;
    
    if (calculation_type != 1 && calculation_type != 2 && calculation_type != 3) {
        cout << "无效选择，程序退出" << endl;
        return 1;
    }
    
    if (calculation_type == 1) {
        // 向量乘向量计算
        cout << "执行向量乘向量计算..." << endl;
        
        // 从文件读取向量
        array<double,16> a, b;
        ifstream fin("vectors_test.txt");
        if (!fin.is_open()) {
            cout << "无法打开 vectors_test.txt 文件" << endl;
            return 1;
        }
        
        for (int i = 0; i < 16; i++) fin >> a[i];
        for (int i = 0; i < 16; i++) fin >> b[i];
        fin.close();
        
        auto start = chrono::high_resolution_clock::now();
        double quantum_result = q_inner(a, b);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        // 经典内积计算
        double classic_result = 0;
        for (int i = 0; i < 16; i++) {
            classic_result += a[i] * b[i];
        }
        
        cout << "计算时间: " << duration.count() << " 微秒" << endl;
        
        // 输出结果到文件
        ofstream fout("result.txt");
        fout << "向量乘向量计算结果:" << endl;
        fout << "输入向量A: ";
        for (int i = 0; i < 16; i++) fout << a[i] << " ";
        fout << endl;
        fout << "输入向量B: ";
        for (int i = 0; i < 16; i++) fout << b[i] << " ";
        fout << endl;
        fout << "量子内积结果: " << fixed << setprecision(6) << quantum_result << endl;
        fout << "经典内积结果: " << fixed << setprecision(6) << classic_result << endl;
        fout << "绝对误差: " << fixed << setprecision(6) << abs(quantum_result - classic_result) << endl;
        fout.close();
        
        // 控制台输出
        cout << "输入向量A: ";
        for (int i = 0; i < 16; i++) cout << a[i] << " ";
        cout << endl;
        cout << "输入向量B: ";
        for (int i = 0; i < 16; i++) cout << b[i] << " ";
        cout << endl;
        cout << "量子内积结果: " << fixed << setprecision(6) << quantum_result << endl;
        cout << "经典内积结果: " << fixed << setprecision(6) << classic_result << endl;
        cout << "绝对误差: " << fixed << setprecision(6) << abs(quantum_result - classic_result) << endl;
        
    } else if (calculation_type == 2) {
        // 矩阵乘矩阵计算
        cout << "执行矩阵乘矩阵计算..." << endl;
        
        // 从文件读取矩阵
        array<array<double,16>,16> matrixA, matrixB;
        ifstream fin1("matrix1.txt");
        ifstream fin2("matrix2.txt");
        
        if (!fin1.is_open() || !fin2.is_open()) {
            cout << "无法打开矩阵输入文件" << endl;
            return 1;
        }
        
        // 读取矩阵A
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                fin1 >> matrixA[i][j];
            }
        }
        
        // 读取矩阵B
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                fin2 >> matrixB[i][j];
            }
        }
        
        fin1.close();
        fin2.close();
        
        // 计算矩阵乘法使用q_inner
        array<array<double,16>,16> quantum_result, classic_result;
        
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                // 提取矩阵A的第i行和矩阵B的第j列
                array<double,16> rowA, colB;
                for (int k = 0; k < 16; k++) {
                    rowA[k] = matrixA[i][k];
                    colB[k] = matrixB[k][j];
                }
                
                // 量子计算
                quantum_result[i][j] = q_inner(rowA, colB);
            }
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                // 经典计算
                classic_result[i][j] = 0;
                for (int k = 0; k < 16; k++) {
                    classic_result[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        
        cout << "计算时间: " << duration.count() << " 微秒" << endl;
        
        // 输出结果到文件
        ofstream fout("result.txt");
        fout << "矩阵乘矩阵计算结果:" << endl;
        fout << "量子矩阵乘法结果:" << endl;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                fout << fixed << setprecision(6) << quantum_result[i][j] << " ";
            }
            fout << endl;
        }
        fout << "经典矩阵乘法结果:" << endl;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                fout << fixed << setprecision(6) << classic_result[i][j] << " ";
            }
            fout << endl;
        }
        fout.close();
        
        // 控制台输出第一行结果作为示例
        cout << "量子矩阵乘法第一行结果: ";
        for (int j = 0; j < 16; j++) {
            cout << fixed << setprecision(6) << quantum_result[0][j] << " ";
        }
        cout << endl;
        cout << "经典矩阵乘法第一行结果: ";
        for (int j = 0; j < 16; j++) {
            cout << fixed << setprecision(6) << classic_result[0][j] << " ";
        }
        cout << endl;
        
    } else if (calculation_type == 3) {
        // 向量乘矩阵计算
        cout << "执行向量乘矩阵计算..." << endl;
        
        // 从文件读取向量和矩阵
        array<double,16> vectorA;
        array<array<double,16>,16> matrixB;
        
        ifstream fin1("vector_ones.txt");
        ifstream fin2("matrix1.txt");
        
        if (!fin1.is_open() || !fin2.is_open()) {
            cout << "无法打开输入文件" << endl;
            return 1;
        }
        
        // 读取向量
        for (int i = 0; i < 16; i++) {
            fin1 >> vectorA[i];
        }
        
        // 读取矩阵
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                fin2 >> matrixB[i][j];
            }
        }
        
        fin1.close();
        fin2.close();
        
        // 计算向量乘矩阵使用q_inner
        array<double,16> quantum_result, classic_result;
        
        auto start = chrono::high_resolution_clock::now();
        for (int j = 0; j < 16; j++) {
            // 提取矩阵B的第j列
            array<double,16> colB;
            for (int k = 0; k < 16; k++) {
                colB[k] = matrixB[k][j];
            }
            
            // 量子计算
            quantum_result[j] = q_inner(vectorA, colB);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        
        for (int j = 0; j < 16; j++) {
            // 经典计算
            classic_result[j] = 0;
            for (int k = 0; k < 16; k++) {
                classic_result[j] += vectorA[k] * matrixB[k][j];
            }
        }
        
        // 输出结果到文件
        ofstream fout("result.txt");
        fout << "向量乘矩阵计算结果:" << endl;
        fout << "输入向量: ";
        for (int i = 0; i < 16; i++) fout << vectorA[i] << " ";
        fout << endl;
        fout << "量子计算结果: ";
        for (int i = 0; i < 16; i++) fout << fixed << setprecision(6) << quantum_result[i] << " ";
        fout << endl;
        fout << "经典计算结果: ";
        for (int i = 0; i < 16; i++) fout << fixed << setprecision(6) << classic_result[i] << " ";
        fout << endl;
        fout.close();
        
        // 控制台输出
        cout << "输入向量: ";
        for (int i = 0; i < 16; i++) cout << vectorA[i] << " ";
        cout << endl;
        cout << "量子计算结果: ";
        for (int i = 0; i < 16; i++) cout << fixed << setprecision(6) << quantum_result[i] << " ";
        cout << endl;
        cout << "经典计算结果: ";
        for (int i = 0; i < 16; i++) cout << fixed << setprecision(6) << classic_result[i] << " ";
        cout << endl;
    }
    
    return 0;
}
