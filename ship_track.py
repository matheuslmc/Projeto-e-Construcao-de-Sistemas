from vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
#Diferentes modelos do YOLOV8
#model = YOLO('yolov8n.pt') #110+ ms
#model = YOLO('yolov8s.pt') #250-550 ms
model = YOLO('yolov8m.pt') #500 - 800 ms
#model = YOLO('yolov8l.pt') #1100+ ms
#model = YOLO('best.pt')

#model = YOLO('yolov8s-seg.pt') #380+ ms

#Diferentes vídeos para teste, com suas resoluções ao lado
#stream = CamGear(source='https://www.youtube.com/watch?v=CubAd2gt4rU', stream_mode=True, logging=True).start() #Navio Royalty Free (1920,1078)
#stream = CamGear(source='https://www.youtube.com/watch?v=tWUIUDd4DgE', stream_mode=True, logging=True).start() #Live do Porto de Santos (1920,1080)
stream = CamGear(source='https://www.youtube.com/watch?v=8WD3lAVvbHo', stream_mode=True, logging=False).start() #Barco Girando (1280,674)
#stream = CamGear(source='https://www.youtube.com/watch?v=a61sshcjOhQ', stream_mode=True, logging=True).start() #Vídeo de teste não relacionado a navios ou barcos (1280,674)



frame_cur = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_size = (1920,1080)
#Inicia um a gravação de um vido
out_vid = cv2.VideoWriter('barco_rodando.mp4',fourcc, 20.0, vid_size)

while frame_cur < 54000:

    frame = stream.read()

    if frame is None:
        print('STREAM END')
        break

    frame_cur += 1

    #CLASSES: {0: 'container', 1: 'cruise', 2: 'fish-b', 3: 'sail boat', 4: 'warship'}
    results = model.predict(frame, stream=True, conf=0.3, classes=[8])

    results = list(results)

    annotated_frame = results[0].plot()
    

    #Anote o número de navios detectados no frame
    cv2.putText(
        annotated_frame,
        ('Navios: ' + str(len(results[0].boxes.cls))),
        (20, annotated_frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
    )

    #Anote o número do frame atual
    cv2.putText(
        annotated_frame,("Quadro atual: "+ str(frame_cur)),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
    )

    #Exiba o frame com anotações
    cv2.imshow('Detection Results', annotated_frame)

    #Registre o frame com anotações no vídeo
    out_vid.write(cv2.resize(annotated_frame, vid_size))


    #Interrompa a execução pressionando a tecla 'Q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print('EARLY STOP')
        break

out_vid.release()

cv2.destroyAllWindows()
stream.stop()