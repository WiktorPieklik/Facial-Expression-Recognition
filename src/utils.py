from simple_chalk import chalk


def info(msg: str) -> None:
    print(chalk.yellow(msg))


def success(msg: str) -> None:
    print(chalk.green(msg))


def error(msg: str) -> None:
    print(chalk.red(msg))


def print_replace(msg: str) -> None:
    print(msg, end='\r')
    print()
