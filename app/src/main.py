import uvicorn
from config import config


def main():
    uvicorn.run(
        "api:app",
        host=config.HOST,
        port=config.PORT,
        timeout_keep_alive=config.KEEP_ALIVE,
    )


if __name__ == "__main__":
    main()
