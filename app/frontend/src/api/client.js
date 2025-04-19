export async function callApi(endpoint, method = 'GET', body = null) {
    const res = await fetch(`/api/${endpoint}`, {
      method,
      headers: body && ! (body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : undefined,
      body: body && !(body instanceof FormData)
        ? JSON.stringify(body)
        : body,
    });
    return res.json();
  }