
## 컴퓨터 비전을 이용한 물체 인식


- 강의개요: [강의계획서](강의계획서.md)
- 준비물: 개발환경이 구축된 개인 노트북
- 질의응답: 강의시간 이외의 질의응답은 본 Repo의 Issues에서 받습니다. 
- 필수사항: 강의에 필요한 개발환경 구축 (1-2주차, 3-4주차)

| 주차 | 강의내용 | 강의 자료 | 실습 자료 | 준비물 | 수고해주시는 분들 | 
|:----:|:----:|:----:|:----:|:----:|:----:|
|  1 | 컴퓨터비전기반 물체 인식 |  [이론](https://www.dropbox.com/s/nixwm5t9s11vwej/CVOR.pdf?dl=0)    | [실습1](1주차-실습1.md), [실습2](1주차-실습2.md), [실습3](1주차-실습3.md) | 개발환경[(Win](1주차-개발환경구축.pdf) / [Mac)](https://github.com/moduPlayGround/ComputerVision-for-VisualRecognition/blob/master/%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD%20MacOS.md)| RCV@Sejong 연구원  |
|  2 | 기계학습기반 물체 인식 |   [이론](https://www.dropbox.com/s/u3w8uqe9hgl2t54/2%EC%A3%BC%EC%B0%A8_%EC%9D%B4%EB%A1%A01.pdf?dl=0)    |  [실습1](2주차-실습1.md), [실습2](2주차-실습2.md)     |  1주차와 동일 |  RCV@Sejong 연구원  |
|  3 | 딥러닝기반 물체 인식-2D | [이론](https://drive.google.com/file/d/1mwR8tnXPMw2lUEbO_TchXdsaobUOsWyw/view?usp=sharing) |  [실습가이드]()    |   | RCV@KAIST 연구원  |
|  4 | 딥러닝기반 물체 인식-3D |   [이론](  https://www.dropbox.com/s/ptz80hymufrj444/%EC%BB%B4%EB%B9%84%EC%A0%84%EA%B0%95%EC%9D%984%EC%A3%BC%EC%B0%A8.pdf?dl=0)  |  ...     |   | RCV@KAIST 연구원  |

## Mac compile
```sh
g++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv@2/2.4.13.7_2/lib/pkgconfig/opencv.pc) -std=c++11 MYFILE.cpp -o MYFILE
```


