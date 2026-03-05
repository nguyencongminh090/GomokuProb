from rich.console import Console
import time
import matplotlib.pyplot as plt
import mplcyberpunk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PyGomo import *
import PyGomo
import matplotlib


CONSOLE = Console()
matplotlib.use('TkAgg')


class ParseMove:
    @staticmethod
    def __valid(move, size=15):
        if len(move) < 2:
            return False

        x = ord(move[0]) - 96
        y = int(move[1:])
        return 0 < x < size + 1 and 0 < y < size + 1

    @staticmethod
    def coord2Num(move, size=15):
        return ord(move[0]) - 97, size - int(move[1:])

    @staticmethod
    def __coord2Num(move):
        return ord(move[0]) - 97, int(move[1:]) - 1

    @staticmethod
    def num2Num(move, size=15):
        return int(move.split(',')[0]), size - 1 - int(move.split(',')[1])

    def get(self, string, numCoord=False, size=15):
        result = []
        while string:
            cur = string[0]
            string = string[1:]
            while string and string[0].isnumeric():
                cur += string[0]
                string = string[1:]
            if self.__valid(cur, size=size):
                result.append(cur if not numCoord else self.__coord2Num(cur))
        return result


class Graph:
    def __init__(self, master):
        self.__fig, self.__ax = plt.subplots()
        self.__ax.get_xaxis().set_visible(False)

        plt.xlim(0)
        plt.ylim(top=100)
        plt.autoscale(axis='x', tight=True)
        plt.grid(axis='y')

        self.__canvas = FigureCanvasTkAgg(self.__fig, master=master)
        self.__canvas.draw()
        self.__canvas.get_tk_widget().pack(fill='both', expand=True)

        self.__chkVal = []
        self.__vefVal = []
        self.__detVal = []

    def addEval(self, ev, typ):
        """
        :param ev: Integer
        :param typ: 0 - check | 1 - verify | 2 - delta
        """
        match typ:
            case 0:
                self.__chkVal.append(ev)
            case 1:
                self.__vefVal.append(ev)
            case 2:
                self.__detVal.append(ev)

    def show(self):
        plt.plot(self.__chkVal, '#2493EF')
        plt.plot(self.__vefVal, '#D44235', marker='o')
        plt.plot(self.__detVal, '#F8BB00')
        mplcyberpunk.add_glow_effects()

    def reset(self):
        self.__chkVal.clear()
        self.__vefVal.clear()
        self.__detVal.clear()


class Game:
    def __init__(self, board):
        self.board = board

    @staticmethod
    def lgs(data):
        for i in range(len(data)):
            rule = [[1, 0], [0, 1], [1, -1], [1, 1]]
            for rx, ry in rule:
                lst = [data[i]]
                if [lst[0][0] + rx * 4, lst[0][1] + ry * 4] not in data or \
                        [lst[0][0] + rx * 3, lst[0][1] + ry * 3] not in data or \
                        [lst[0][0] + rx * 2, lst[0][1] + ry * 2] not in data or \
                        [lst[0][0] + rx * 1, lst[0][1] + ry * 1] not in data \
                        or [lst[0][0] + rx * 5, lst[0][1] + ry * 5] in data or [lst[0][0] - rx, lst[0][1] - ry] in data:
                    continue
                else:
                    return True
        return False

    def is_win(self):
        data = []
        for i in self.board:
            data.append([i[0], i[1]])
        db = self.lgs(data[::2])
        dw = self.lgs(data[1::2])
        return True in (db, dw)


def calc(pos: list, controller, tm: int) -> int:
    with CONSOLE.status("[bold green]Calculating..."):
        return analyze(pos, controller, tm)


def waitFunc(n, onClose):
    def waitTime():
        start = time.perf_counter()
        while time.perf_counter() - start < n:
            continue
        onClose()

    def decorator(func):
        def wrapper(*arg, **kwargs):
            Thread(target=waitTime, daemon=True).start()
            return func(*arg, **kwargs)
        return wrapper
    return decorator


def analyze(pos: list, controller: Controller, tm: int) -> float:
    def onClose():
        controller.protocol().stop()

    @waitFunc(tm, onClose)
    def getInfo():
        return controller.get('move')
    
    controller.protocol().setPos([f'{move[0]},{move[1]}' for move in pos])
    enMove: PyGomo.Move = getInfo()
    print(f'[-] Depth: {enMove.info["depth"]}')
    # print(f'[-] Ev   : {enMove.info["ev"].toWinrate(1)}%')
    return enMove.info['ev'].toWinrate(1)


def turnOnEngine(controller: Controller):
    config = {
                'timeout_match': -1,
                'timeout_turn': -1,
                'game_type': 1,
                'rule': 1,
                'usedatabase': 1}

    controller.setConfig(config)
    controller.setTimeMatch(9999 * 1000)
    controller.setTimeLeft(9999 * 1000)
    if controller.isReady():
        print('Engine loaded')
    else:
        print("Can't load engine!")
        return


def main():
    pos       = str(input('Pos          : '))
    startFrom = int(input('Start From   :                 [Must be > 1]\rStart From   : ')) - 1
    side      = int(input('Side         :                 [Black = 0 | White = 1]\rSide         : '))
    timePM    = int(input('Time per move:                 [seconds]\rTime per move: '))

    pos = ParseMove().get(pos, 1)

    # Tkinter
    root = tk.Tk()
    root.title("Graph")
    graph = Graph(root)

    # Engine
    engine     = Engine(r'/media/ngmint/Data/Programming/Python/Personal/GomoProgC-main/Engine/rapfi')
    protocol   = Protocol()
    controller = Controller(engine, protocol)
    turnOnEngine(controller)

    for i, move in enumerate(pos):
        if i >= startFrom and i % 2 == side:
            print(f'Finished [{round(i/len(pos) * 100, 2)}%]')
            if Game(pos[:i]).is_win() or Game(pos[:i + 1]).is_win():
                break
            print('--> CHECK')
            check  = calc(pos[:i], controller, timePM)
            # print(f'[+] Check: {check}')
            graph.addEval(check, 0)

            print('--> VERIFY')
            verify = 100 - calc(pos[:i + 1], controller, timePM)
            # print(f'[+] Verify: {verify}')
            graph.addEval(verify, 1)

            delta  = abs(verify - check)
            # print(f'[+] Delta: {delta}')
            graph.addEval(delta, 2)

    controller.quit()

    graph.show()
    root.mainloop()


if __name__ == '__main__':
    main()
