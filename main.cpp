

//#include "RollingGuidanceFilter.h"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
// 
//using namespace std;
//using namespace cv;
//
//
//float* createHist(Mat src){
//	src.convertTo(src, CV_8U);
//	Mat res = src.clone();
//	cvtColor(src,res,CV_BGR2Lab);
//	
//	float R[16]= {0};
//	for(int x=0 ; x<res.rows ; x++){
//		for(int y=0 ; y<res.cols ; y++){
//			int n = res.at<Vec3b>(x,y)[0] / 16 ;
//			R[n]++;
//		}
//	}
//	for(int i=0 ; i<16 ; i++){
//		R[i] /= (src.rows*src.cols);
//		R[i] = 1- R[i];
//		printf("%f " ,  R[i]);
//	}
//	printf("\n");
//	return R;
//}
//
//float segment(Mat res){
//	res.convertTo(res, CV_8U);
//	Mat n = res.clone();
//	cvtColor(res,n,CV_BGR2Lab);
//	float sum = 0;
//	for(int x=0 ; x<res.rows ; x++){
//		for(int y=0 ; y<res.cols ; y++){
//			sum += n.at<Vec3b>(x,y)[0];
//			printf("%d ", n.at<Vec3b>(x,y)[0]);
//		}
//		printf("\n");
//	}
//	
//	printf("******************************************************************");
//	sum = sum/(res.rows * res.cols);
//	return sum;
//}
//
///* generate guide image as per the iteration number */
//Mat getGuideImage(int niter ,int i, Mat img , float sigma){
//	
//	// if niter=1 guide image is the gaussian output of the image
//	if(niter==1){
//		Mat guide = img.clone();
//		GaussianBlur( img, guide , Size( 5, 5 ), 0, 0 );
//		return guide;
//	}
//	// Output of niter iteration is the guide image for the next iteration
//	else{
//		Mat guide = RollingGuidanceFilter::filter(img,sigma,25.5,niter-1);
//		return guide;
//	}
//}
//
///* the main function */
//int main(){
//	
//	String name = "./imgs/e.png";
//
//	Mat img = imread(name); // read the input image
//	
//	if(img.empty()){
//		printf("No such file.\n");
//		getchar();
//		exit(1);
//	}
//
//	clock_t startTime = clock();
//	float R[] = {0.5,1,1.5,2,2.5,3,3.5,4};    //spatial tolerance
//	float J[8];
//	Mat M[8];
//	Mat M_final;
//
//	for(int i = 0 ; i<8 ; i++){
//		//Mat guide = img.clone();
//		//guide = getGuideImage(1,i,img,R[i]);
//		//imwrite("./imgs/guide0"+to_string(i+1) +".jpg",guide);
//		Mat res = RollingGuidanceFilter::filter(img,R[i],25.5,4);
//		imwrite("./imgs/roll"+to_string(i+1) +".jpg",res);
//		Laplacian(res,res,3);
//		printf("Elapsed Time: %d ms\n",clock()-startTime);
//		imwrite("./imgs/laplace"+to_string(i+1) +".jpg",res);
//		float* r8 = createHist(res);
//
//		res.convertTo(res, CV_8U);
//		Mat n = res.clone();
//		cvtColor(res,n,CV_BGR2Lab);
//		Mat fi[16];
//		//M[i] = Mat::zeros(res.rows,res.cols,CV_8UC3);
//		for(int j = 0; j<16 ; j++){
//			fi[j] = n.clone();
//			for(int x = 0 ; x<fi[j].rows ; x++){
//				for(int y = 0 ; y<fi[j].cols ; y++){
//					if(n.at<Vec3b>(x,y)[0]/16 == j)fi[j].at<Vec3b>(x,y)[0]= 255;
//					else fi[j].at<Vec3b>(x,y)[0]= 0;
//				}
//			}
//			cvtColor(fi[j],fi[j],CV_Lab2BGR);
//			imwrite("./imgs/fi_"+to_string(i)+to_string(j)+".jpg",fi[j]);
//			GaussianBlur( fi[j], fi[j],Size(0,0), R[i]);
//			cvtColor(fi[j],fi[j],CV_BGR2Lab);
//			if (j==0){
//				M[i] = fi[j].clone();
//			}
//			else{
//				for(int x = 0 ; x<fi[j].rows ; x++){
//					for(int y = 0 ; y<fi[j].cols ; y++){
//						M[i].at<Vec3b>(x,y)[0] += (((float)fi[j].at<Vec3b>(x,y)[0]/255.0)*r8[j])*255;
//						M[i].at<Vec3b>(x,y)[1] += fi[j].at<Vec3b>(x,y)[1];
//						M[i].at<Vec3b>(x,y)[2] += fi[j].at<Vec3b>(x,y)[2];
//					}
//				}
//			}
//		}
//		if(i == 0){
//			M_final = M[i].clone();
//		}
//		else{
//			for(int x = 0; x<img.rows ; x++){
//				for(int y = 0; y<img.cols ; y++){
//					M_final.at<Vec3b>(x,y)[0] += M[i].at<Vec3b>(x,y)[0];
//					M_final.at<Vec3b>(x,y)[1] += M[i].at<Vec3b>(x,y)[1];
//					M_final.at<Vec3b>(x,y)[2] += M[i].at<Vec3b>(x,y)[2];
//				}
//			}
//		}
//		cvtColor(M[i],M[i],CV_Lab2BGR);
//		imwrite("./imgs/m_"+to_string(i)+".jpg",M[i]);
//		
//	}
//	cvtColor(M_final,M_final,CV_Lab2BGR);
//	imwrite("./imgs/MF.jpg",M_final);
//	
//	/*
//	imshow("res",img);
//	printf("||||||||||||||||||||||||||||||||||||||||||||||||||||||j = %d\n",j);
//	waitKey(0);
//	*/
//	
//	imshow("res",img);
//	waitKey(0);
//	return 0;
//}



#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"
#include "RollingGuidanceFilter.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

float* createHist(Mat src){
	//src.convertTo(src, CV_8UC1);
	
	float* R = new float[16];
	for(int x=0 ; x<16 ; x++){
		R[x]=0;
	}
	for(int x=0 ; x<src.rows ; x++){
		for(int y=0 ; y<src.cols ; y++){
			int n = src.at<uchar>(x,y)/ 16 ;
			R[n]++;
		}
	}
	for(int i=0 ; i<16 ; i++){
		R[i] /= (src.rows*src.cols);
		R[i] = 1- R[i];
		printf("%f " ,  R[i]);
	}
	printf("\n");
	return R;
}

int max_pixel(int** src, int rows , int cols){
	int max = -1;
	printf("**************************************************************max %d %d\n" , rows , cols );
	for(int x = 0 ; x<rows ; x++){
		for(int y = 0 ; y<cols ; y++){
			if( max < src[x][y] ){
				max = src[x][y];
			}
		}
	}
	printf("%d max value" , max);
	return max;
}

int min_pixel(int** src , int rows , int cols){
	int min = 100000;
	printf("**************************************************************min %d %d\n" , rows , cols );
	for(int x = 0 ; x<rows ; x++){
		for(int y = 0 ; y<cols ; y++){
			if( min > src[x][y] ){
				min = src[x][y];
			}
		}
	}
	printf("%d min value" , min);
	return min;
}
int** scale_image(int** src , int min , int max , int rows , int cols){
	int** out ;
	printf("**************************************************************min %d %d\n" , rows , cols );
	out = (int**)malloc(rows*sizeof(int));
	printf("check point 1");
	for(int i = 0 ; i<rows ; i++){
		out[i] = (int*)malloc(cols*sizeof(int));
	}
	printf("check point 2");
	for(int x = 0 ; x<rows ; x++){
		for(int y = 0 ; y<cols ; y++){
			out[x][y] = ((src[x][y] - min)*255)/(max-min);
		}
	}
	printf("check point 3");
	return out;
}

//((uint8)((B == 255) ? B:min(255, ((A << 8 ) / (255 - B)))))
void pencil_sketch(Mat src , String name){
	/*
	Mat neg = Mat::zeros(src.rows , src.cols , src.type());
	for(int x = 0 ; x<src.rows ; x++){
		for(int y = 0 ; y<src.cols ; y++){
			neg.at<uchar>(x,y) = 255 - src.at<uchar>(x,y);
		}
	}

	Mat blur = Mat::zeros(src.rows , src.cols , src.type());
	GaussianBlur( src, blur,Size(0,0), 5);

	Mat blend = Mat::zeros(src.rows , src.cols , src.type());
	for(int x = 0 ; x<src.rows ; x++){
		for(int y = 0 ; y<src.cols ; y++){
			//blend.at<uchar>(x,y) = min(255,(neg.at<uchar>(x,y)+blur.at<uchar>(x,y)));
			if(neg.at<uchar>(x,y)==255){
				blend.at<uchar>(x,y) = 255;
			}
			else{
				blend.at<uchar>(x,y) = min(255,((blur.at<uchar>(x,y) << 8) / (255-neg.at<uchar>(x,y))));
			}
		}
	}

	imwrite("./imgs_b/sketch.jpg",blend);
	*/

	Mat out = Mat::zeros(src.rows , src.cols , src.type()); 
	//float ker[6][6] = {{1, 1, 1, 1, 1, 1},{1, 1, 1, 1, 1, 1},{1, 1, -8, -8, 1, 1},{1, 1, -8, -8, 1, 1},{1, 1, 1, 1, 1, 1},{1, 1, 1, 1, 1, 1}};
	Mat ker = Mat::zeros(6,6,CV_32FC1);
	for(int x = 0 ; x<6 ; x++){
		for(int y = 0; y<6 ; y++){
			ker.at<float>(x,y) = 1;
		}
	}
	for(int x = 2 ; x<4 ; x++){
		for(int y = 2 ; y<4 ; y++){
			ker.at<float>(x,y) = -8;
		}
	}
	ker = ker/6;

	filter2D(src, out, -1, ker);
	
	Mat neg = Mat::zeros(src.rows , src.cols , src.type());
	for(int x = 0 ; x<src.rows ; x++){
		for(int y = 0 ; y<src.cols ; y++){
			neg.at<uchar>(x,y) = 255 - out.at<uchar>(x,y);
		}
	}
	
	imwrite(name,neg);
}

Mat Map_post_processing(Mat src){
	Mat frame = RollingGuidanceFilter::filter(src,0.01,0.5,4);
	return frame;
}


/* generate guide image as per the iteration number 
Mat getGuideImage(int niter ,int i, Mat img , float sigma){
	
	// if niter=1 guide image is the gaussian output of the image
	if(niter==1){
		Mat guide = img.clone();
		GaussianBlur( img, guide , Size( 5, 5 ), 0, 0 );
		return guide;
	}
	// Output of niter iteration is the guide image for the next iteration
	else{
		Mat guide = RollingGuidanceFilter::filter(img,sigma,25.5,niter-1);
		return guide;
	}
}
*/

void detailEnhancement(Mat img)
{
//	Mat img = imread("parrot.jpg");
	Mat laplacianSharpening, unsharpSharpening;

    // Laplacian sharpening
	float laplacianBoostFactor = 1.2; // Controls the amount of Laplacian Sharpening
	Mat kern = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5*laplacianBoostFactor, -1, 0, -1, 0); // The filtering mask for Laplacian sharpening
	filter2D(img.clone(), laplacianSharpening, img.depth(), kern, Point(-1, -1)); // Applies the masking operator to the image

	// Unsharp mask sharpening
	Mat blur;
	GaussianBlur(img.clone(), blur, Size(5, 5), 0.67, 0.67);
	Mat unsharpMask = img.clone() - blur; // Compute the unsharp mask
	threshold(unsharpMask.clone(), unsharpMask, 20, 255, THRESH_TOZERO);
	float unsharpBoostFactor = 9; // Controls the amount of Unsharp Mask Sharpening
	unsharpSharpening = img.clone() + (unsharpMask * unsharpBoostFactor);
	/*
	imshow("Original Image", img);
	imshow("Laplacian Sharpened Image", laplacianSharpening);
	imshow("Unsharp Masking Sharpening", unsharpSharpening);
	*/
	imwrite("./imgs_b/Laplacian_Sharp_Boost_1_point_2.jpg", laplacianSharpening);
	imwrite("./imgs_b/Unsharp_Sharp_Boost_9_Threshold_20.jpg", unsharpSharpening);
}

int computeOutput(int x, int r1, int s1, int r2, int s2)
{
    float result;
    if(0 <= x && x <= r1){
        result = s1/r1 * x;
    }else if(r1 < x && x <= r2){
        result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
    }else if(r2 < x && x <= 255){
        result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
    }
    return (int)result;
}

void ContrastStretch(Mat image)
{
    Mat new_image = image.clone();

    int r1, s1, r2, s2;
    //cout<<"Enter r1: "<<endl; cin>>r1;
    //cout<<"Enter s1: "<<endl; cin>>s1;
    //cout<<"Enter r2: "<<endl; cin>>r2;
    //cout<<"Enter s2: "<<endl; cin>>s2;
	r1 = 150;
	r2 = 200;
	s1 = 50;
	s2 = 240;
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){
            //(int c = 0; c < 3; c++){
                int output = computeOutput(image.at<uchar>(y,x), r1, s1, r2, s2);
                new_image.at<uchar>(y,x) = saturate_cast<uchar>(output);
            //}
        }
    }

	imwrite("./imgs_b/contrast.jpg" , new_image);

  /*  namedWindow("Original Image", 1);
    imshow("Original Image", image);

    namedWindow("New Image", 1);
    imshow("New Image", new_image);

    waitKey();*/

   // return 0;
}

/*
void Cartoonize(){
	Mat img1 = imread("./imgs_b/pics/c1.jpg");
	Mat img2 = imread("./imgs_b/pics/c2.jpg");
	Mat img3 = imread("./imgs_b/pics/c3.jpg");
	Mat img4 = imread("./imgs_b/pics/c4.jpg");
	int o = 8;
	int iter = 4;
	float i1  = 25.5;
	Mat res = RollingGuidanceFilter::filter(img1,o,i1,iter);
	imwrite("./imgs_b/roll" + to_string(1) + ".jpg" , res);
	Mat res1 = RollingGuidanceFilter::filter(img2,o,i1,iter);
	imwrite("./imgs_b/roll" + to_string(2) + ".jpg" , res1);
	Mat res2 = RollingGuidanceFilter::filter(img3,o,i1,iter);
	imwrite("./imgs_b/roll" + to_string(3) + ".jpg" , res2);
	Mat res3 = RollingGuidanceFilter::filter(img4,o,i1,iter);
	imwrite("./imgs_b/roll" + to_string(4) + ".jpg" , res3);

	printf("done");
}
*/

void Cartoonize(int ch){
	/*
	String name = "./imgs_b/cartoon.png";
	Mat color = imread(name,1);

	if(color.empty()){
		printf("No such file.\n");
		getchar();
		exit(1);
	}
	
	Mat bgr[3];
	split(color,bgr);
	Mat img[3];
	Mat res[3];
	float R[] = {0.5,1,1.5};//,2.5,3,3.5,4};    //spatial tolerance
	Mat M_final[3],M_final_f;
	Mat output[3],output_f;
	Mat frame[3];
	Mat temp_f;
	Mat* inp1 = new Mat[3];

	for(int channels = 0 ; channels<3 ; channels++){
		img[channels] = bgr[channels].clone();
		clock_t startTime = clock();
		
		float J[8];
		Mat M[8];
		
		printf("Parul1\n");

		for(int i = 0 ; i<3 ; i++){
			res[channels] = RollingGuidanceFilter::filter(img[channels],R[i],25.5,4);
			imwrite("./imgs_b/roll_color" +to_string(channels) +"_"+to_string(i) + ".jpg" , res[channels]);
			Laplacian(res[channels],res[channels],3);
			//res.convertTo(res, CV_8UC1);
		
			printf("Elapsed Time: %d ms\n",clock()-startTime);

			imwrite("./imgs_b/laplace" +to_string(channels) +"_"+to_string(i+1) +".jpg",res[channels]);
			
			float* r8 = createHist(res[channels]);

			Mat fi[16];
			
			for(int j = 0; j<16 ; j++){
				fi[j] = Mat::zeros(img[channels].rows , img[channels].cols , img[channels].type());//res.clone();
			
				for(int x = 0 ; x<fi[j].rows ; x++){
					for(int y = 0 ; y<fi[j].cols ; y++){
						if(res[channels].at<uchar>(x,y)/16 == j)fi[j].at<uchar>(x,y)= 255;
						else fi[j].at<uchar>(x,y)= 0;
					}
				}
				if( j==0 ){
					fi[j] = 255 - fi[j];
				}
				imwrite("./imgs_b/fi_"+ to_string(channels)+"_"+to_string(i)+"_"+to_string(j)+".jpg",fi[j]);
				GaussianBlur( fi[j], fi[j],Size(0,0), R[i]);
			
				if (j==0){
					M[i] = fi[j].clone();
					M[i] = M[i]*0;
				}
				else{
					for(int x = 0 ; x<fi[j].rows ; x++){
						for(int y = 0 ; y<fi[j].cols ; y++){
							M[i].at<uchar>(x,y) += (((float)fi[j].at<uchar>(x,y)/255.0)*r8[j])*255;
						}
					}
				}
			}
			printf("Parul2\n");
			if(i == 0){
				M_final[channels] = M[i].clone();
				M_final[channels] = M_final[channels]/16;
				//M_final = M_final*0;
			}
			else{
				M_final[channels] += M[i]*((i+1)/8.0);
			}
			imwrite("./imgs_b/m_"+to_string(channels)+"_"+to_string(i)+".jpg",M[i]);
		
		}

		imwrite("./imgs_b/MF"+ to_string(channels)+".jpg",M_final[channels]);
		Mat inp;
		Mat guide;
		M_final[channels].convertTo(inp,CV_MAKETYPE(CV_32F,img[channels].channels()));
		img[channels].convertTo(guide,CV_MAKETYPE(CV_32F,img[channels].channels()));
		printf("Parul3\n");
		output[channels] = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
		imwrite("./imgs_b/output.jpg" , output[channels]); // M map

		frame[channels] = Map_post_processing(output[channels]);
		frame[channels].convertTo(frame[channels],CV_MAKETYPE(img[channels].depth(),img[channels].channels()));
		imwrite("./imgs_b/output_ref"+to_string(channels)+".jpg" , 255-frame[channels]); // M map refined
		Mat temp1 = Mat::zeros(img[channels].rows , img[channels].cols , img[channels].type());
	
		for( int pro=0 ; pro<10 ; pro++){
			Mat inp;
			Mat guide;
			img[channels].convertTo(inp,CV_MAKETYPE(CV_32F,img[channels].channels()));
			frame[channels].convertTo(guide,CV_MAKETYPE(CV_32F,img[channels].channels()));
			img[channels] = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
		}
	
		printf("Parul4\n");
		Mat guide1;
		img[channels].convertTo(inp1[channels],CV_MAKETYPE(CV_32F,img[channels].channels()));
		frame[channels].convertTo(guide1,CV_MAKETYPE(CV_32F,img[channels].channels()));
		printf("Parul5\n");
		for( int pro=0 ; pro<5 ; pro++){
			inp1[channels] = RollingGuidanceFilter::bilateralPermutohedral(inp1[channels],guide1,1.0,5.0);
		}
		imwrite("./imgs_b/temp1"+to_string(channels)+".jpg" , inp1[channels]);
		printf("Parul6\n");
		inp1[channels].convertTo(inp1[channels],CV_MAKETYPE(img[channels].depth(),img[channels].channels()));
		//merge(inp1,3,temp_f);
		printf("Parul7\n");
		//imwrite("./imgs_b/final.jpg",temp_f);
		printf("**********!!!!!!!!!!!!!!!!!!**********-------11111*******************!!!!!!!!!!*");
	}
	*/
	//-------------------------------------------------------------------------------------------------------------------
	/*
	String name = "./imgs_b/cartoon.png";
	Mat color = imread(name,0);

	if(color.empty()){
		printf("No such file.\n");
		getchar();
		exit(1);
	}
	clock_t startTime = clock();
	float R[] = {0.5,1,1.5};//,2.5,3,3.5,4};    //spatial tolerance
	float J[8];
	Mat M[8];
	Mat M_final;

	for(int i = 0 ; i<3 ; i++){
		Mat res = RollingGuidanceFilter::filter(color,R[i],25.5,4);
		imwrite("./imgs_b/roll_color" + to_string(i) + ".jpg" , res);
		Laplacian(res,res,3);
		//res.convertTo(res, CV_8UC1);
		
		printf("Elapsed Time: %d ms\n",clock()-startTime);

		imwrite("./imgs_b/laplace"+to_string(i+1) +".jpg",res);
		float* r8 = createHist(res);

		Mat fi[16];
		
		for(int j = 0; j<16 ; j++){
			fi[j] = Mat::zeros(img.rows , img.cols , img.type());//res.clone();
			
			for(int x = 0 ; x<fi[j].rows ; x++){
				for(int y = 0 ; y<fi[j].cols ; y++){
					if(res.at<uchar>(x,y)/16 == j)fi[j].at<uchar>(x,y)= 255;
					else fi[j].at<uchar>(x,y)= 0;
				}
			}
			if( j==0 ){
				fi[j] = 255 - fi[j];
			}
			imwrite("./imgs_b/fi_"+to_string(i)+"_"+to_string(j)+".jpg",fi[j]);
			GaussianBlur( fi[j], fi[j],Size(0,0), R[i]);
			
			if (j==0){
				M[i] = fi[j].clone();
				M[i] = M[i]*0;
			}
			else{
				for(int x = 0 ; x<fi[j].rows ; x++){
					for(int y = 0 ; y<fi[j].cols ; y++){
						M[i].at<uchar>(x,y) += (((float)fi[j].at<uchar>(x,y)/255.0)*r8[j])*255;
					}
				}
			}
		}
		
		if(i == 0){
			M_final = M[i].clone();
			M_final = M_final/16;
			//M_final = M_final*0;
		}
		else{
			M_final += M[i]*((i+1)/8.0);
		}
		imwrite("./imgs_b/m_"+to_string(i)+".jpg",M[i]);
		
	}

	imwrite("./imgs_b/MF.jpg",M_final);
	Mat inp;
	Mat guide;
	M_final.convertTo(inp,CV_MAKETYPE(CV_32F,img.channels()));
	img.convertTo(guide,CV_MAKETYPE(CV_32F,img.channels()));

	Mat output = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
	imwrite("./imgs_b/output.jpg" , output); // M map

	Mat frame = Map_post_processing(output);
	frame.convertTo(frame,CV_MAKETYPE(img.depth(),img.channels()));
	imwrite("./imgs_b/output_ref.jpg" , 255-frame); // M map refined
	Mat temp1 = Mat::zeros(img.rows , img.cols , img.type());
	
	for( int pro=0 ; pro<10 ; pro++){
		Mat inp;
		Mat guide;
		img.convertTo(inp,CV_MAKETYPE(CV_32F,img.channels()));
		frame.convertTo(guide,CV_MAKETYPE(CV_32F,img.channels()));
		img = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
	}
	
	Mat inp1;
	Mat guide1;
	img.convertTo(inp1,CV_MAKETYPE(CV_32F,img.channels()));
	frame.convertTo(guide1,CV_MAKETYPE(CV_32F,img.channels()));
		
	for( int pro=0 ; pro<5 ; pro++){
		inp1 = RollingGuidanceFilter::bilateralPermutohedral(inp1,guide1,1.0,5.0);
	}
	imwrite("./imgs_b/temp1.jpg" , inp1);
	
	inp1.convertTo(inp1,CV_MAKETYPE(img.depth(),img.channels()));
*/
}

/* the main function */

int main(){
	

	Mat* temp = new Mat[3];
	for(int ch  = 0 ; ch<3 ; ch++){
		temp[ch]= imread("./imgs_b/imgs_c/temp1_"+to_string(ch)+".jpg",0);
	}

	Mat final_img ; 
	merge(temp,3,final_img); 
	imwrite("./imgs_b/imgs_c/final.jpg",final_img);
	//Cartoonize();

	/*
	int ch = 2;
	String name = "./imgs_b/imgs_c/cartoon"+to_string(ch)+".jpg";
	
	Mat clr = imread(name,1);
	if(clr.empty()){
		printf("No such file.\n");
		getchar();
		exit(1);
	}
	
	Mat bgr[3];
	split(clr,bgr);
	imwrite("./imgs_b/imgs_c/cartoon0.jpg",bgr[0]);
	imwrite("./imgs_b/imgs_c/cartoon1.jpg",bgr[1]);
	imwrite("./imgs_b/imgs_c/cartoon2.jpg",bgr[2]);
	*/
	/*
	Mat img = imread(name,0); // read the input image
	detailEnhancement(img);
	//imwrite("./imgs_b/temp0.jpg" , img);
	pencil_sketch(img , "./imgs_b/imgs_c/sketch_inp"+to_string(ch)+".jpg");
	Mat color = imread(name,1);

	if(img.empty()){
		printf("No such file.\n");
		getchar();
		exit(1);
	}
	
	Mat edge_lapl ;
	Mat edge_canny;
	Laplacian(img,edge_lapl,1000);
	Canny( img, edge_canny, 50, 150, 3);
	imwrite("./imgs_b/imgs_c/edge_lapl"+to_string(ch)+".jpg" , 255-edge_lapl);
	imwrite("./imgs_b/imgs_c/edge_canny"+to_string(ch)+".jpg" , 255-edge_canny);

	clock_t startTime = clock();
	float R[] = {0.5,1,1.5};//,2.5,3,3.5,4};    //spatial tolerance
	float J[8];
	Mat M[8];
	Mat M_final;

	for(int i = 0 ; i<3 ; i++){
		int scaleValue = 5;
		Mat res = RollingGuidanceFilter::filter(img,R[i],25.5,4);
		imwrite("./imgs_b/roll" + to_string(i) + ".jpg" , res);

		//pencil_sketch(res);

		Mat enis = (img - res)*scaleValue + res;
		imwrite("./imgs_b/enh" + to_string(i) + ".jpg" , enis);

		Mat resc = RollingGuidanceFilter::filter(color,R[i],25.5,4);
		imwrite("./imgs_b/roll_color" + to_string(i) + ".jpg" , resc);
		Mat enisc = (color - resc)*scaleValue + 1*resc;
		imwrite("./imgs_b/enh_color" + to_string(i) + ".jpg" , enisc);
		

		Laplacian(res,res,3);
		res.convertTo(res, CV_8UC1);

		
		printf("Elapsed Time: %d ms\n",clock()-startTime);

		imwrite("./imgs_b/imgs_c/laplace"+to_string(ch)+"_"+to_string(i+1) +".jpg",res);
		float* r8 = createHist(res);

		Mat fi[16];
		
		for(int j = 0; j<16 ; j++){
			fi[j] = Mat::zeros(img.rows , img.cols , img.type());//res.clone();
			
			for(int x = 0 ; x<fi[j].rows ; x++){
				for(int y = 0 ; y<fi[j].cols ; y++){
					if(res.at<uchar>(x,y)/16 == j)fi[j].at<uchar>(x,y)= 255;
					else fi[j].at<uchar>(x,y)= 0;
				}
			}
			if( j==0 ){
				fi[j] = 255 - fi[j];
			}
			imwrite("./imgs_b/fi_"+to_string(i)+"_"+to_string(j)+".jpg",fi[j]);
			GaussianBlur( fi[j], fi[j],Size(0,0), R[i]);
			
			if (j==0){
				M[i] = fi[j].clone();
				M[i] = M[i]*0;
			}
			else{
				for(int x = 0 ; x<fi[j].rows ; x++){
					for(int y = 0 ; y<fi[j].cols ; y++){
						M[i].at<uchar>(x,y) += (((float)fi[j].at<uchar>(x,y)/255.0)*r8[j])*255;
					}
				}
			}
		}
		
		if(i == 0){
			M_final = M[i].clone();
			M_final = M_final/16;
			//M_final = M_final*0;
		}
		else{
			M_final += M[i]*((i+1)/8.0);
		}
		imwrite("./imgs_b/imgs_c/m"+to_string(ch)+"_"+to_string(i)+".jpg",M[i]);
		
	}

	imwrite("./imgs_b/imgs_c/MF"+to_string(ch)+".jpg",M_final);
	//detailEnhance(Mat src, Mat dst, float sigma_s=10, float sigma_r=0.15f)

	//Mat output = RollingGuidanceFilter::filter(M_final,1.5,25.5,4);
	Mat inp;
	Mat guide;
	M_final.convertTo(inp,CV_MAKETYPE(CV_32F,img.channels()));
	img.convertTo(guide,CV_MAKETYPE(CV_32F,img.channels()));

	Mat output = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
	imwrite("./imgs_b/imgs_c/output"+to_string(ch)+".jpg" , output); // M map

	Mat frame = Map_post_processing(output);
	frame.convertTo(frame,CV_MAKETYPE(img.depth(),img.channels()));
	//Mat eq_hist;
	//equalizeHist(255-frame,eq_hist);
	imwrite("./imgs_b/imgs_c/output_ref"+to_string(ch)+".jpg" , 255-frame); // M map refined
	ContrastStretch(255-frame);
	Mat temp1 = Mat::zeros(img.rows , img.cols , img.type());
	
	//for( int pro=0 ; pro<10 ; pro++){
	//	Mat inp;
	//	Mat guide;
	//	img.convertTo(inp,CV_MAKETYPE(CV_32F,img.channels()));
	//	frame.convertTo(guide,CV_MAKETYPE(CV_32F,img.channels()));
	//	img = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
	//}
	
	Mat inp1;
	Mat guide1;
	img.convertTo(inp1,CV_MAKETYPE(CV_32F,img.channels()));
	frame.convertTo(guide1,CV_MAKETYPE(CV_32F,img.channels()));
		
	for( int pro=0 ; pro<5 ; pro++){
		//img = RollingGuidanceFilter::bilateralPermutohedral(inp,guide,1.0,5.0);
		inp1 = RollingGuidanceFilter::bilateralPermutohedral(inp1,guide1,1.0,5.0);
	}
	

	imwrite("./imgs_b/imgs_c/temp1_"+to_string(ch)+".jpg" , inp1);


	inp1.convertTo(inp1,CV_MAKETYPE(img.depth(),img.channels()));
	pencil_sketch(inp1 , "./imgs_b/imgs_c/sketch_otp_"+to_string(ch)+".jpg");
	*/

	//imwrite("./imgs_b/temp1.jpg" , img);
	//img.convertTo(inp,CV_MAKETYPE(CV_8UC1,1));

	/*Mat frame_o;
	GaussianBlur(frame , frame_o , Size(0,0) , 3);
	addWeighted(frame , 0.5 , frame_o , -0.5 , 0 , frame);

	imwrite("./imgs_b/output_refo.jpg" , frame_o); // M map
	*/
	/*
	Mat I_basic = RollingGuidanceFilter::filter(img,1,25.5,4);
	imwrite("./imgs_b/I_basic.jpg" , I_basic);
	int **IoM_arr;
	IoM_arr = (int**)malloc(img.rows*sizeof(int));
	for(int i = 0 ; i<img.rows ; i++){
		IoM_arr[i] = (int*)malloc(img.cols*sizeof(int));
	}
	
	for(int x = 0 ; x<img.rows ; x++){
		for(int y = 0 ; y<img.cols ; y++){
			IoM_arr[x][y] = img.at<uchar>(x,y) * output.at<uchar>(x,y);
		}
	}
	IoM_arr = scale_image(IoM_arr , min_pixel(IoM_arr,img.rows,img.cols) , max_pixel(IoM_arr,img.rows,img.cols) , img.rows , img.cols);
	
	printf("\n\n\n\n");
	printf("check point 4");
	Mat IoM_ = Mat::zeros(img.rows, img.cols,img.type() );
	for(int d = 0 ; d<img.rows ; d++){
		for(int e=0 ; e<img.cols ; e++){
			IoM_.at<uchar>(d,e) = IoM_arr[d][e];
		}
	}
	imwrite("./imgs_b/IoM.jpg" , IoM_);
	printf("check point 5");
	Mat I_det = I_basic - IoM_;
	imwrite("./imgs_b/I_det.jpg" , I_det);
	printf("check point 6");
	int** MoID_arr ;

	MoID_arr = (int**)malloc(img.rows*sizeof(int));
	
	for(int i = 0 ; i<img.rows ; i++){
		MoID_arr[i] = (int*)malloc(img.cols*sizeof(int));
	}
	printf("check point 7");
	
	for(int x = 0 ; x<img.rows ; x++){
		for(int y = 0 ; y<img.cols ; y++){
			MoID_arr[x][y] = I_det.at<uchar>(x,y) * output.at<uchar>(x,y);
		}
	}
	printf("check point 8");
	MoID_arr = scale_image(MoID_arr , min_pixel(MoID_arr,img.rows,img.cols) , max_pixel(MoID_arr,img.rows,img.cols),img.rows,img.cols);
	printf("check point 9");
	Mat MoID_ = Mat::zeros(img.rows, img.cols,img.type());
	for(int d = 0 ; d<img.rows ; d++){
		for(int e=0 ; e<img.cols ; e++){
			MoID_.at<uchar>(d,e) = MoID_arr[d][e];
		}
	}
	//------------------------------------------------------
	float lambda = 1;
	int** enI_arr ;

	enI_arr = (int**)malloc(img.rows*sizeof(int));
	
	for(int i = 0 ; i<img.rows ; i++){
		enI_arr[i] = (int*)malloc(img.cols*sizeof(int));
	}
	printf("check point 10");
	
	for(int x = 0 ; x<img.rows ; x++){
		for(int y = 0 ; y<img.cols ; y++){
			enI_arr[x][y] = I_basic.at<uchar>(x,y) + lambda * MoID_.at<uchar>(x,y) ;//I_det.at<uchar>(x,y) * output.at<uchar>(x,y);
		}
	}
	printf("check point 11");
	enI_arr = scale_image(enI_arr , min_pixel(enI_arr,img.rows,img.cols) , max_pixel(enI_arr,img.rows,img.cols),img.rows,img.cols);
	printf("check point 12");
	Mat enI_ = Mat::zeros(img.rows, img.cols,img.type());
	for(int d = 0 ; d<img.rows ; d++){
		for(int e=0 ; e<img.cols ; e++){
			enI_.at<uchar>(d,e) = enI_arr[d][e];
		}
	}
	//------------------------------------------------------
	
	//Mat enI = I_basic + lambda*MoID_;
	//enI = scale_image(enI , min_pixel(enI) , max_pixel(enI));
	//Canny( output, output, 50, 150, 3);
	//for(int x = 0 ; x< output.rows; x++){
	//	for(int y = 0; y<output.cols; y++){
	//		output.at<uchar>(x,y) = 255 - output.at<uchar>(x,y);
	//	}
	//}
	//imwrite("./imgs_b/outputt.jpg" , output);
	
	imwrite("./imgs_b/MoID.jpg" , MoID_);
	
	imwrite("./imgs_b/enI.jpg" , enI_);
	
	Mat exper = RollingGuidanceFilter::filter(img-M_final,2,25.5,4);*/
	//pencil_sketch(exper);
	
	
	
	//detailEnhancement(img);

	//waitKey(0);
	return 0;
	//ERROR TO BE REMOVED: WHEN MAX==MIN
}

