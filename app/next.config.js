const isDev = process.env.NODE_ENV === "development";

const baseUrl = isDev ? "http://127.0.0.1:8001" : "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  rewrites: async () => {
    return [
      {
        source: `/api/py/:path*`,
        destination: `${baseUrl}/api/py/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
