import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from core_utils.logger import get_logger

# Initialize structured logging from core_utils
logger = get_logger("TerminalAPI")


class APIHandler(BaseHTTPRequestHandler):
    """
    Minimal, zero-dependency HTTP server to act as the AEGIS Terminal API gateway.
    Handles basic health checks and provides metrics for the Prometheus scraper.
    """

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"status": "healthy", "service": "AEGIS Terminal API"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            # Basic system health metrics formatted for Prometheus scraper
            metrics = (
                "# HELP aegis_api_health System health status\n"
                "# TYPE aegis_api_health gauge\n"
                "aegis_api_health 1\n"
            )
            self.wfile.write(metrics.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def log_message(self, format_str, *args):
        # Route standard HTTP request logging through core logging utility
        logger.info(format_str % args)


def run():
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, APIHandler)
    logger.info("AEGIS Terminal API starting on port 8000...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    logger.info("Stopping AEGIS Terminal API...")


if __name__ == "__main__":
    run()
