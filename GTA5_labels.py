class GTA5Labels():
    def __init__(self):
        
        road = {"ID":1, "color":(128, 64, 128)}
        sidewalk = {"ID":2, "color":(244, 35, 232)}
        building = {"ID":3, "color":(70, 70, 70)}
        wall = {"ID":4, "color":(102, 102, 156)}
        fence = {"ID":5, "color":(190, 153, 153)}
        pole = {"ID":6, "color":(153, 153, 153)}
        light = {"ID":7, "color":(250, 170, 30)}
        sign = {"ID":8, "color":(220, 220, 0)}
        vegetation = {"ID":9, "color":(107, 142, 35)}
        terrain = {"ID":10, "color":(152, 251, 152)}
        sky = {"ID":11, "color":(70, 130, 180)}
        person = {"ID":12, "color":(220, 20, 60)}
        rider = {"ID":13, "color":(255, 0, 0)}
        car = {"ID":14, "color":(0, 0, 142)}
        truck = {"ID":15, "color":(0, 0, 70)}
        bus = {"ID":16, "color":(0, 60, 100)}
        train = {"ID":17, "color":(0, 80, 100)}
        motocycle = {"ID":18, "color":(0, 0, 230)}
        bicycle = {"ID":19, "color":(119, 11, 32)}

        self.list_ = [road,sidewalk,building,wall,fence,pole,light,sign,vegetation,
                 terrain,sky,person,rider,car,truck,bus,train,motocycle,bicycle,]
    def support_id_list(self):
        ret = [label["ID"] for label in self.list_]
        return ret